"""
Optional TensorRT acceleration for TransFuser++ inference.

Usage (agent): set TRT=1. The first run exports the network to ONNX and builds a TensorRT engine next to the
checkpoint (e.g. model_0030_0.fp32.trt, ~1.5 min); later runs just load the cached engine. Engines are specific to
the GPU and TensorRT version; delete the .trt file to rebuild.

Only the outputs the driving agent needs are exported (target speed, checkpoint/path, bounding-box head, and the
waypoint branches when they exist). The auxiliary heads (semantic, depth, BEV semantic) are training-only and are
pruned from the engine.

TrtNet is a drop-in replacement for LidarCenterNet.forward(): it returns the same 10-tuple, with None for the pruned
auxiliary outputs, and forwards convert_features_to_bb_metric() to the original network.
"""
import os
import time

import tensorrt as trt
import torch
import torch.nn as nn
import torch.nn.functional as F

_LOGGER = trt.Logger(trt.Logger.WARNING)

# Positions inside the LidarCenterNet.forward() output tuple that the agent consumes.
# (pred_wp, pred_target_speed, pred_checkpoint, pred_semantic, pred_bev_semantic, pred_depth,
#  pred_bounding_box, attention_weights, pred_wp_1, selected_path)
_IDX_WP, _IDX_TS, _IDX_CKPT, _IDX_BB, _IDX_WP1, _IDX_PATH = 0, 1, 2, 6, 8, 9


class _ExportWrapper(nn.Module):
  """Flattens the outputs the agent needs into a plain tuple of tensors, so the graph can be exported to ONNX."""

  def __init__(self, net):
    super().__init__()
    self.net = net
    self.output_names = []
    self.num_bb = 0

  def forward(self, rgb, lidar_bev, target_point, ego_vel, command):
    out = self.net.forward(rgb=rgb, lidar_bev=lidar_bev, target_point=target_point, ego_vel=ego_vel, command=command)
    flat, names = [], []
    for name, idx in (('pred_wp', _IDX_WP), ('pred_target_speed', _IDX_TS), ('pred_checkpoint', _IDX_CKPT),
                      ('pred_wp_1', _IDX_WP1), ('selected_path', _IDX_PATH)):
      if out[idx] is not None:
        flat.append(out[idx])
        names.append(name)
    if out[_IDX_BB] is not None:
      self.num_bb = len(out[_IDX_BB])
      for i, t in enumerate(out[_IDX_BB]):
        if t is not None:  # some heads (e.g. velocity/brake) can be disabled
          flat.append(t)
          names.append(f'bb_{i}')
    self.output_names = names
    return tuple(flat)


class _StaticAdaptivePool(nn.Module):
  """Export-only replacement for nn.AdaptiveAvgPool2d. The ONNX tracer loses the (static) spatial size after the
  interpolations in the fusion blocks, so we replay the input sizes recorded in an eager pass as fixed kernels."""

  def __init__(self, output_size, in_shapes):
    super().__init__()
    self.output_size, self.in_shapes, self.calls = output_size, in_shapes, 0

  def forward(self, x):
    h, w = self.in_shapes[self.calls % len(self.in_shapes)]
    self.calls += 1
    oh, ow = self.output_size
    if h % oh or w % ow:
      raise NotImplementedError(f'adaptive pool {h}x{w} -> {oh}x{ow} is not an integer ratio')
    return F.avg_pool2d(x, (h // oh, w // ow))


def _swap_adaptive_pools(net, wrapper, inputs):
  """Replace every AdaptiveAvgPool2d by a static pool. Returns an undo function."""
  recorded, hooks, targets = {}, [], []
  for name, m in net.named_modules():
    if isinstance(m, nn.AdaptiveAvgPool2d):
      recorded[name] = []
      hooks.append(m.register_forward_hook(lambda mod, i, o, n=name: recorded[n].append(tuple(i[0].shape[2:]))))
      targets.append((name, m))
  with torch.no_grad():
    wrapper(**inputs)
  for h in hooks:
    h.remove()
  originals = []
  for name, m in targets:
    parent_name, _, attr = name.rpartition('.')
    parent = net.get_submodule(parent_name) if parent_name else net
    size = m.output_size if isinstance(m.output_size, tuple) else (m.output_size, m.output_size)
    if size == (1, 1):
      continue  # global pooling is exported fine
    setattr(parent, attr, _StaticAdaptivePool(size, recorded[name]))
    originals.append((parent, attr, m))

  def undo():
    for parent, attr, m in originals:
      setattr(parent, attr, m)

  return undo


def _example_inputs(config, device):
  # rgb is cropped by the agent before the network, so use the cropped size.
  import transfuser_utils as t_u
  cropped = t_u.crop_array(config, torch.zeros((config.camera_height, config.camera_width, 3)).numpy())
  rgb = torch.zeros((1, 3, cropped.shape[0], cropped.shape[1]), device=device)
  lidar_bev = torch.zeros((1, 1 + int(config.use_ground_plane), config.lidar_resolution_height,
                           config.lidar_resolution_width), device=device)
  return dict(rgb=rgb,
              lidar_bev=lidar_bev,
              target_point=torch.tensor([[10.0, 0.0]], device=device),
              ego_vel=torch.tensor([[5.0]], device=device),
              command=torch.tensor([[0, 0, 0, 0, 1, 0]], device=device).float())


def _export_onnx(net, config, onnx_path, device):
  wrapper = _ExportWrapper(net).eval()
  inputs = _example_inputs(config, device)
  undo = _swap_adaptive_pools(net, wrapper, inputs)
  # The fused eval-mode attention kernel has no ONNX symbolic; use the regular decomposed path for export.
  torch.backends.mha.set_fastpath_enabled(False)
  try:
    with torch.no_grad():
      wrapper(**inputs)  # populates output_names
      torch.onnx.export(wrapper,
                        tuple(inputs.values()),
                        onnx_path,
                        input_names=list(inputs.keys()),
                        output_names=wrapper.output_names,
                        opset_version=17,
                        do_constant_folding=True,
                        dynamo=False)
  finally:
    torch.backends.mha.set_fastpath_enabled(True)
    undo()
  return list(inputs.keys()), wrapper.output_names


def _build_engine(onnx_path, engine_path):
  builder = trt.Builder(_LOGGER)
  network = builder.create_network()
  parser = trt.OnnxParser(network, _LOGGER)
  with open(onnx_path, 'rb') as f:
    if not parser.parse(f.read()):
      errors = '\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors))
      raise RuntimeError(f'TensorRT failed to parse the ONNX model:\n{errors}')
  cfg = builder.create_builder_config()
  cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
  start = time.time()
  serialized = builder.build_serialized_network(network, cfg)
  if serialized is None:
    raise RuntimeError('TensorRT engine build failed')
  with open(engine_path, 'wb') as f:
    f.write(serialized)
  print(f'[trt] built {engine_path} in {time.time() - start:.0f}s')


def _torch_dtype(trt_dtype):
  return {trt.float32: torch.float32, trt.float16: torch.float16, trt.int32: torch.int32,
          trt.int64: torch.int64}[trt_dtype]


class TrtNet:
  """Runs a cached TensorRT engine and mimics LidarCenterNet.forward / convert_features_to_bb_metric."""

  def __init__(self, net, config, checkpoint_path, device):
    if getattr(config, 'tp_attention', False) or getattr(config, 'two_tp_input', False):
      raise NotImplementedError('TRT backend does not support tp_attention / two_tp_input models yet')
    self.net = net  # kept for convert_features_to_bb_metric
    self.device = device
    stem = os.path.splitext(checkpoint_path)[0]
    engine_path = f'{stem}.fp32.trt'
    if not os.path.exists(engine_path):
      onnx_path = f'{stem}.onnx'
      print('[trt] exporting ONNX ...')
      _export_onnx(net, config, onnx_path, device)
      print('[trt] building engine (one-time, a few minutes) ...')
      _build_engine(onnx_path, engine_path)
    with open(engine_path, 'rb') as f:
      self.engine = trt.Runtime(_LOGGER).deserialize_cuda_engine(f.read())
    self.context = self.engine.create_execution_context()
    self.stream = torch.cuda.Stream(device=device)
    # The engine holds all the weights the forward pass needs. Free the PyTorch copy from the GPU (it is only kept for
    # CPU-side helpers): on an 8 GB card next to the CARLA server this avoids spilling into shared memory.
    self.net.cpu()
    torch.cuda.empty_cache()
    self.input_names, self.output_names = [], []
    for i in range(self.engine.num_io_tensors):
      name = self.engine.get_tensor_name(i)
      if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
        self.input_names.append(name)
      else:
        self.output_names.append(name)
    # get_bboxes() takes 7 head outputs; heads that were disabled stay None.
    self.num_bb = 7 if any(n.startswith('bb_') for n in self.output_names) else 0
    self.outputs = {
        n: torch.empty(tuple(self.engine.get_tensor_shape(n)),
                       dtype=_torch_dtype(self.engine.get_tensor_dtype(n)),
                       device=device) for n in self.output_names
    }
    for n, t in self.outputs.items():
      self.context.set_tensor_address(n, t.data_ptr())

  def forward(self, rgb, lidar_bev, target_point, ego_vel, command, target_point_next=None):
    feeds = dict(rgb=rgb, lidar_bev=lidar_bev, target_point=target_point, ego_vel=ego_vel, command=command)
    keep = []  # keep contiguous copies alive until the stream is done
    for name in self.input_names:
      t = feeds[name].contiguous().float()
      keep.append(t)
      self.context.set_input_shape(name, tuple(t.shape))
      self.context.set_tensor_address(name, t.data_ptr())
    # Run on a dedicated stream (TensorRT adds extra synchronisation on the default stream) and order it
    # against PyTorch's current stream before and after.
    current = torch.cuda.current_stream(self.device)
    self.stream.wait_stream(current)
    if not self.context.execute_async_v3(self.stream.cuda_stream):
      raise RuntimeError('TensorRT execution failed')
    current.wait_stream(self.stream)
    o = self.outputs
    bb = [o[f'bb_{i}'].float() if f'bb_{i}' in o else None for i in range(self.num_bb)]
    f32 = lambda k: o[k].float() if k in o else None
    # Same layout as LidarCenterNet.forward(); the clones keep results valid across calls.
    clone = lambda t: t.clone() if t is not None else None
    return (clone(f32('pred_wp')), clone(f32('pred_target_speed')), clone(f32('pred_checkpoint')), None, None, None,
            tuple(t.clone() if t is not None else None for t in bb) if self.num_bb else None, None, clone(f32('pred_wp_1')), clone(f32('selected_path')))

  def convert_features_to_bb_metric(self, bb_predictions):
    return self.net.convert_features_to_bb_metric(bb_predictions)

  def __getattr__(self, name):
    # Everything else the agent uses (control_pid_direct, speed_histogram, ...) lives on the original network.
    if name == 'net':
      raise AttributeError(name)
    return getattr(self.net, name)
