import torch
import gc


def get_gpu_info():
    if not torch.cuda.is_available():
        return None
    
    device = torch.device('cuda')
    total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    reserved_mem = torch.cuda.memory_reserved(device) / (1024**3)
    allocated_mem = torch.cuda.memory_allocated(device) / (1024**3)
    free_mem = total_mem - reserved_mem
    
    return {
        'device_name': torch.cuda.get_device_name(device),
        'total_memory_gb': total_mem,
        'reserved_memory_gb': reserved_mem,
        'allocated_memory_gb': allocated_mem,
        'free_memory_gb': free_mem,
        'memory_utilization': reserved_mem / total_mem * 100,
    }


def is_low_vram_gpu(total_mem_gb=None):
    if total_mem_gb is None:
        if not torch.cuda.is_available():
            return False
        total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    return total_mem_gb <= 6.0


def get_recommended_batch_size(input_size, model_name='standard', total_mem_gb=None):
    if total_mem_gb is None:
        if not torch.cuda.is_available():
            return 32
        total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    base_batch = {
        'tiny': 64,
        'small': 48,
        'standard': 32,
        'large': 16,
        'mobilenet': 48,
    }
    
    base = base_batch.get(model_name, 32)
    
    if total_mem_gb >= 12:
        multiplier = 2.0
    elif total_mem_gb >= 8:
        multiplier = 1.5
    elif total_mem_gb >= 6:
        multiplier = 1.0
    elif total_mem_gb >= 4:
        multiplier = 0.75
    else:
        multiplier = 0.5
    
    size_factor = 128 / input_size
    
    batch_size = int(base * multiplier * size_factor)
    
    batch_size = max(4, min(batch_size, 128))
    
    batch_size = (batch_size // 4) * 4
    if batch_size == 0:
        batch_size = 4
    
    return batch_size


def get_recommended_input_size(total_mem_gb=None):
    if total_mem_gb is None:
        if not torch.cuda.is_available():
            return 128
        total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    if total_mem_gb >= 8:
        return 224
    elif total_mem_gb >= 6:
        return 192
    elif total_mem_gb >= 4:
        return 160
    else:
        return 128


def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()


def setup_gpu_optimizations():
    if not torch.cuda.is_available():
        return {'success': False, 'message': 'CUDA不可用'}
    
    try:
        device = torch.device('cuda')
        total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        torch.cuda.set_per_process_memory_fraction(0.95, device=device)
        
        if total_mem <= 6:
            os_environ_set = False
            try:
                import os
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
                os_environ_set = True
            except:
                pass
            
            return {
                'success': True,
                'message': f'低显存优化已启用 ({total_mem:.1f}GB)',
                'low_vram': True,
                'total_memory_gb': total_mem,
                'cudnn_benchmark': True,
                'memory_fraction': 0.95,
                'os_environ_set': os_environ_set,
            }
        else:
            return {
                'success': True,
                'message': f'GPU优化已启用 ({total_mem:.1f}GB)',
                'low_vram': False,
                'total_memory_gb': total_mem,
                'cudnn_benchmark': True,
                'memory_fraction': 0.95,
            }
    except Exception as e:
        return {'success': False, 'message': f'GPU优化失败: {str(e)}'}


def get_gradient_accumulation_steps(batch_size, target_effective_batch=32):
    if batch_size >= target_effective_batch:
        return 1
    
    steps = target_effective_batch // batch_size
    if target_effective_batch % batch_size > 0:
        steps += 1
    
    return min(steps, 8)


def estimate_memory_usage(batch_size, input_size, model_name='standard', num_classes=10):
    model_params = {
        'tiny': 0.5,
        'small': 3,
        'standard': 10,
        'large': 25,
        'mobilenet': 2,
    }
    
    params_mb = model_params.get(model_name, 10)
    
    activations_mb = batch_size * input_size * input_size * 3 * 4 / (1024**2)
    activations_mb *= 50
    
    gradients_mb = params_mb * 4
    
    optimizer_mb = params_mb * 8
    
    total_mb = params_mb + activations_mb + gradients_mb + optimizer_mb
    
    overhead = 1.5
    total_mb *= overhead
    
    return total_mb / 1024


def check_memory_available(required_gb, total_mem_gb=None):
    if not torch.cuda.is_available():
        return True, 'CPU模式'
    
    if total_mem_gb is None:
        total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    free_gb = get_gpu_info()['free_memory_gb']
    
    if free_gb >= required_gb:
        return True, f'显存充足 (需要{required_gb:.1f}GB, 可用{free_gb:.1f}GB)'
    else:
        return False, f'显存不足 (需要{required_gb:.1f}GB, 可用{free_gb:.1f}GB)'


def auto_adjust_for_gpu(batch_size, input_size, model_name, num_classes, log_fn=None):
    if not torch.cuda.is_available():
        return batch_size, input_size, 1, 'CPU模式，无需调整'
    
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    free_gb = get_gpu_info()['free_memory_gb']
    
    required_mem = estimate_memory_usage(batch_size, input_size, model_name, num_classes)
    
    if free_gb >= required_mem * 1.3:
        if log_fn:
            log_fn(f'[GPU] 显存充足: 需要{required_mem:.1f}GB, 可用{free_gb:.1f}GB')
        return batch_size, input_size, 1, '显存充足'
    
    if free_gb >= required_mem * 1.1:
        if log_fn:
            log_fn(f'[GPU] 显存够用: 需要{required_mem:.1f}GB, 可用{free_gb:.1f}GB')
        return batch_size, input_size, 1, '显存够用'
    
    original_batch = batch_size
    
    while batch_size > 4 and free_gb < required_mem * 1.1:
        batch_size = batch_size // 2
        required_mem = estimate_memory_usage(batch_size, input_size, model_name, num_classes)
    
    if batch_size != original_batch:
        accum_steps = max(1, original_batch // batch_size)
        if log_fn:
            log_fn(f'[GPU优化] 批量 {original_batch} → {batch_size}, 梯度累积 {accum_steps} 步')
        clear_gpu_cache()
        return batch_size, input_size, accum_steps, f'显存优化 (批量:{batch_size}, 累积:{accum_steps}步)'
    
    clear_gpu_cache()
    return batch_size, input_size, 1, '显存检查通过'


def format_memory_info():
    info = get_gpu_info()
    if info is None:
        return 'GPU不可用'
    
    return (
        f"GPU: {info['device_name']}\n"
        f"总显存: {info['total_memory_gb']:.2f} GB\n"
        f"已分配: {info['allocated_memory_gb']:.2f} GB\n"
        f"已预留: {info['reserved_memory_gb']:.2f} GB\n"
        f"可用: {info['free_memory_gb']:.2f} GB\n"
        f"利用率: {info['memory_utilization']:.1f}%"
    )
