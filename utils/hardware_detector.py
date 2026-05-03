import os
import sys
import time
import gc
import platform
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


@dataclass
class HardwareInfo:
    cpu_name: str
    cpu_cores: int
    cpu_threads: int
    ram_total_gb: float
    ram_available_gb: float
    gpu_name: str
    gpu_total_memory_gb: float
    gpu_available_memory_gb: float
    gpu_compute_capability: str
    platform_info: str


@dataclass
class BenchmarkResult:
    model_name: str
    input_size: int
    batch_size: int
    time_per_batch_ms: float
    samples_per_second: float
    memory_used_gb: float
    success: bool
    error_msg: str = ""


@dataclass
class OptimalConfig:
    model_name: str
    input_size: int
    batch_size: int
    num_workers: int
    use_amp: bool
    accum_steps: int
    estimated_time_per_epoch_min: float
    confidence: str


def _get_windows_version() -> str:
    if platform.system() != "Windows":
        return f"{platform.system()} {platform.release()} ({platform.machine()})"
    
    try:
        import subprocess
        result = subprocess.run(
            ['cmd', '/c', 'ver'],
            capture_output=True, text=True, timeout=5
        )
        ver_output = result.stdout.strip()
        
        if "10.0.22" in ver_output or "10.0.26" in ver_output:
            win_version = "Windows 11"
        elif "10.0.1" in ver_output:
            win_version = "Windows 10"
        else:
            try:
                import winreg
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                    r"SOFTWARE\Microsoft\Windows NT\CurrentVersion")
                product_name, _ = winreg.QueryValueEx(key, "ProductName")
                win_version = product_name
                winreg.CloseKey(key)
            except Exception:
                if platform.version().startswith("10.0.22") or platform.version().startswith("10.0.26"):
                    win_version = "Windows 11"
                else:
                    win_version = f"Windows {platform.release()}"
        
        arch = platform.machine()
        if arch == "AMD64":
            arch = "x64"
        
        return f"{win_version} ({arch})"
        
    except Exception:
        return f"Windows {platform.release()} ({platform.machine()})"


def get_hardware_info() -> HardwareInfo:
    cpu_name = "Unknown CPU"
    cpu_cores = 1
    cpu_threads = 1
    
    if HAS_PSUTIL:
        cpu_cores = psutil.cpu_count(logical=False) or 1
        cpu_threads = psutil.cpu_count(logical=True) or 1
        ram_total = psutil.virtual_memory().total / (1024**3)
        ram_available = psutil.virtual_memory().available / (1024**3)
    else:
        import multiprocessing
        cpu_cores = multiprocessing.cpu_count() or 1
        cpu_threads = cpu_cores
        ram_total = 8.0
        ram_available = 4.0
    
    if platform.system() == "Windows":
        try:
            import subprocess
            result = subprocess.run(
                ['wmic', 'cpu', 'get', 'name'],
                capture_output=True, text=True, timeout=5
            )
            lines = result.stdout.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line and line != 'Name':
                    cpu_name = line
                    break
        except Exception:
            cpu_name = f"CPU ({cpu_cores} cores)"
    else:
        try:
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if 'model name' in line:
                        cpu_name = line.split(':')[1].strip()
                        break
        except Exception:
            cpu_name = f"CPU ({cpu_cores} cores)"
    
    gpu_name = "No GPU"
    gpu_total_memory = 0.0
    gpu_available_memory = 0.0
    gpu_compute_cap = "N/A"
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_available_memory = gpu_total_memory
        
        try:
            torch.cuda.set_device(0)
            torch.cuda.empty_cache()
            free, total = torch.cuda.mem_get_info(0)
            gpu_available_memory = free / (1024**3)
        except Exception:
            pass
        
        try:
            cap = torch.cuda.get_device_capability(0)
            gpu_compute_cap = f"{cap[0]}.{cap[1]}"
        except Exception:
            pass
    
    platform_info = _get_windows_version()
    
    return HardwareInfo(
        cpu_name=cpu_name,
        cpu_cores=cpu_cores,
        cpu_threads=cpu_threads,
        ram_total_gb=ram_total,
        ram_available_gb=ram_available,
        gpu_name=gpu_name,
        gpu_total_memory_gb=gpu_total_memory,
        gpu_available_memory_gb=gpu_available_memory,
        gpu_compute_capability=gpu_compute_cap,
        platform_info=platform_info,
    )


def _create_dummy_dataset(num_samples: int, input_size: int, num_classes: int = 10) -> Dataset:
    class DummyDataset(Dataset):
        def __init__(self, num_samples, input_size, num_classes):
            self.num_samples = num_samples
            self.input_size = input_size
            self.num_classes = num_classes
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            images = torch.randn(3, self.input_size, self.input_size)
            labels = torch.randint(0, self.num_classes, (1,)).item()
            return images, labels
    
    return DummyDataset(num_samples, input_size, num_classes)


def benchmark_model_forward(
    model: nn.Module,
    input_size: int,
    batch_size: int,
    device: torch.device,
    num_iterations: int = 10,
    warmup: int = 3,
    use_amp: bool = False,
    num_classes: int = 10,
) -> Tuple[float, float, bool, str]:
    model = model.to(device)
    model.train()
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp) if use_amp and device.type == 'cuda' else None
    
    dummy_input = torch.randn(batch_size, 3, input_size, input_size, device=device)
    dummy_target = torch.randint(0, num_classes, (batch_size,), device=device)
    
    try:
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(dummy_input)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
        
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        
        for _ in range(num_iterations):
            optimizer.zero_grad()
            
            if use_amp and device.type == 'cuda':
                with torch.amp.autocast(device.type):
                    output = model(dummy_input)
                    loss = criterion(output, dummy_target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(dummy_input)
                loss = criterion(output, dummy_target)
                loss.backward()
                optimizer.step()
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        avg_time_ms = (total_time / num_iterations) * 1000
        
        memory_used = 0.0
        if device.type == 'cuda':
            memory_used = torch.cuda.max_memory_allocated() / (1024**3)
        
        return avg_time_ms, memory_used, True, ""
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                gc.collect()
            return 0, 0, False, "OOM"
        return 0, 0, False, str(e)
    except Exception as e:
        return 0, 0, False, str(e)


def find_max_batch_size(
    model_name: str,
    input_size: int,
    device: torch.device,
    num_classes: int = 86,
    start_batch: int = 8,
    max_batch: int = 256,
    use_amp: bool = True,
    log_fn: Callable = None,
    total_gpu_memory_gb: float = 0,
) -> Tuple[int, List[BenchmarkResult]]:
    from model import create_model
    
    results = []
    
    if log_fn:
        log_fn(f"[显存探测] 模型:{model_name}, 输入尺寸:{input_size}")
    
    def test_batch(batch_size: int) -> Tuple[float, float, bool, str]:
        try:
            model = create_model(model_name, num_classes, input_size)
            time_ms, mem_gb, success, error = benchmark_model_forward(
                model, input_size, batch_size, device,
                num_iterations=3, warmup=1, use_amp=use_amp, num_classes=num_classes
            )
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                gc.collect()
            return time_ms, mem_gb, success, error
        except Exception as e:
            return 0, 0, False, str(e)
    
    low, high = start_batch, max_batch
    max_successful_batch = 0
    last_mem_gb = 0
    
    time_ms, mem_gb, success, error = test_batch(low)
    if not success:
        if log_fn:
            log_fn(f"  批量 {low} 失败: {error}")
        return 0, results
    
    max_successful_batch = low
    last_mem_gb = mem_gb
    results.append(BenchmarkResult(
        model_name=model_name, input_size=input_size, batch_size=low,
        time_per_batch_ms=time_ms, samples_per_second=(low/time_ms)*1000 if time_ms > 0 else 0,
        memory_used_gb=mem_gb, success=True
    ))
    
    if total_gpu_memory_gb > 0 and mem_gb > 0:
        max_possible = int(total_gpu_memory_gb / mem_gb * low * 0.85)
        high = min(high, max_possible)
    
    while low <= high:
        mid = (low + high) // 2
        mid = (mid // 4) * 4
        
        if mid <= max_successful_batch:
            low = mid + 4
            continue
        
        if log_fn:
            log_fn(f"  测试批量 {mid}...")
        
        time_ms, mem_gb, success, error = test_batch(mid)
        
        if success:
            max_successful_batch = mid
            last_mem_gb = mem_gb
            results.append(BenchmarkResult(
                model_name=model_name, input_size=input_size, batch_size=mid,
                time_per_batch_ms=time_ms, samples_per_second=(mid/time_ms)*1000 if time_ms > 0 else 0,
                memory_used_gb=mem_gb, success=True
            ))
            if log_fn:
                log_fn(f"    ✓ 成功: 显存 {mem_gb:.2f}GB")
            low = mid + 4
            
            if total_gpu_memory_gb > 0 and mem_gb > total_gpu_memory_gb * 0.9:
                break
        else:
            if log_fn:
                log_fn(f"    ✗ {error}")
            high = mid - 4
    
    if max_successful_batch > 0 and max_successful_batch < max_batch:
        for extra in [max_successful_batch + 4, max_successful_batch + 8]:
            if extra <= max_batch and extra > max_successful_batch:
                if total_gpu_memory_gb > 0 and last_mem_gb > 0:
                    estimated = last_mem_gb * (extra / max_successful_batch) * 1.1
                    if estimated > total_gpu_memory_gb * 0.92:
                        continue
                
                if log_fn:
                    log_fn(f"  精细测试 {extra}...")
                
                time_ms, mem_gb, success, error = test_batch(extra)
                if success:
                    max_successful_batch = extra
                    last_mem_gb = mem_gb
                    results.append(BenchmarkResult(
                        model_name=model_name, input_size=input_size, batch_size=extra,
                        time_per_batch_ms=time_ms, samples_per_second=(extra/time_ms)*1000 if time_ms > 0 else 0,
                        memory_used_gb=mem_gb, success=True
                    ))
                    if log_fn:
                        log_fn(f"    ✓ 成功")
                else:
                    break
    
    if log_fn:
        log_fn(f"  → 最大批量: {max_successful_batch}")
    
    return max_successful_batch, results


def benchmark_dataloader(
    data_dir: str,
    input_size: int,
    batch_size: int,
    num_workers: int,
    num_batches: int = 20,
) -> float:
    from dataset_loader import get_transforms
    from torchvision import datasets
    
    transform = get_transforms(input_size, is_train=True)
    
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    
    start_time = time.perf_counter()
    batch_count = 0
    
    for _ in loader:
        batch_count += 1
        if batch_count >= num_batches:
            break
    
    total_time = time.perf_counter() - start_time
    
    try:
        del loader
    except Exception:
        pass
    
    return total_time / num_batches if batch_count > 0 else float('inf')


def find_optimal_num_workers(
    data_dir: str,
    input_size: int,
    batch_size: int,
    max_workers: int = 8,
    log_fn: Callable = None,
) -> Tuple[int, Dict[int, float]]:
    if log_fn:
        log_fn(f"[数据加载测试] 输入尺寸:{input_size}, 批量:{batch_size}")
    
    workers_to_test = [0, 1, 2, 4]
    if max_workers >= 8:
        workers_to_test.append(8)
    
    results = {}
    
    for num_workers in workers_to_test:
        if num_workers > max_workers:
            continue
        
        if log_fn:
            log_fn(f"  测试 num_workers={num_workers}...")
        
        try:
            avg_time = benchmark_dataloader(
                data_dir, input_size, batch_size, num_workers, num_batches=15
            )
            results[num_workers] = avg_time
            
            if log_fn:
                log_fn(f"    平均每批: {avg_time*1000:.1f}ms")
                
        except Exception as e:
            if log_fn:
                log_fn(f"    失败: {str(e)}")
            results[num_workers] = float('inf')
    
    if results:
        optimal = min(results, key=results.get)
        return optimal, results
    
    return 0, {0: float('inf')}


def benchmark_all_models(
    input_sizes: List[int],
    device: torch.device,
    num_classes: int = 86,
    use_amp: bool = True,
    log_fn: Callable = None,
) -> Dict[str, Dict[int, BenchmarkResult]]:
    from model import create_model
    
    models = ['tiny', 'small', 'standard', 'large', 'mobilenet']
    results = {model: {} for model in models}
    
    if log_fn:
        log_fn("[模型速度基准测试]")
    
    for model_name in models:
        for input_size in input_sizes:
            if log_fn:
                log_fn(f"  测试 {model_name} @ {input_size}x{input_size}...")
            
            try:
                model = create_model(model_name, num_classes, input_size)
                
                batch_size = 8
                time_ms, mem_gb, success, error = benchmark_model_forward(
                    model, input_size, batch_size, device,
                    num_iterations=10, warmup=3, use_amp=use_amp, num_classes=num_classes
                )
                
                if success:
                    samples_per_sec = (batch_size / time_ms) * 1000 if time_ms > 0 else 0
                    results[model_name][input_size] = BenchmarkResult(
                        model_name=model_name,
                        input_size=input_size,
                        batch_size=batch_size,
                        time_per_batch_ms=time_ms,
                        samples_per_second=samples_per_sec,
                        memory_used_gb=mem_gb,
                        success=True
                    )
                    if log_fn:
                        log_fn(f"    ✓ {time_ms:.1f}ms/批, {samples_per_sec:.0f}样本/秒, 显存:{mem_gb:.2f}GB")
                else:
                    results[model_name][input_size] = BenchmarkResult(
                        model_name=model_name,
                        input_size=input_size,
                        batch_size=batch_size,
                        time_per_batch_ms=0,
                        samples_per_second=0,
                        memory_used_gb=0,
                        success=False,
                        error_msg=error
                    )
                    if log_fn:
                        log_fn(f"    ✗ {error}")
                
                del model
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                    gc.collect()
                    
            except Exception as e:
                if log_fn:
                    log_fn(f"    ✗ 异常: {str(e)}")
    
    return results


def run_full_benchmark(
    data_dir: str,
    num_classes: int = 86,
    log_fn: Callable = None,
    progress_fn: Callable = None,
) -> Tuple[HardwareInfo, OptimalConfig]:
    if log_fn:
        log_fn("=" * 60)
        log_fn("[1/3] 检测硬件信息...")
    
    hw_info = get_hardware_info()
    
    if log_fn:
        log_fn(f"  CPU: {hw_info.cpu_name}")
        log_fn(f"  核心数: {hw_info.cpu_cores}核/{hw_info.cpu_threads}线程")
        log_fn(f"  内存: {hw_info.ram_total_gb:.1f}GB (可用: {hw_info.ram_available_gb:.1f}GB)")
        log_fn(f"  GPU: {hw_info.gpu_name}")
        if hw_info.gpu_total_memory_gb > 0:
            log_fn(f"  显存: {hw_info.gpu_total_memory_gb:.1f}GB (可用: {hw_info.gpu_available_memory_gb:.1f}GB)")
        log_fn(f"  平台: {hw_info.platform_info}")
    
    if progress_fn:
        progress_fn(1, 3)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    
    total_images = 0
    if data_dir and os.path.isdir(data_dir):
        try:
            from pathlib import Path
            image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
            for subdir in Path(data_dir).iterdir():
                if subdir.is_dir():
                    total_images += len([f for f in subdir.iterdir() if f.suffix.lower() in image_exts])
        except Exception:
            total_images = 1000
    
    if total_images == 0:
        total_images = 1000
    
    if log_fn:
        log_fn("")
        log_fn(f"[2/3] 显存探测 (数据集: {total_images}张, {num_classes}类)...")
    
    best_model = 'standard'
    best_input_size = 128
    best_batch_size = 16
    
    candidate_models = ['resnet18', 'large', 'standard']
    candidate_input_sizes = [192, 128]
    
    if hw_info.gpu_total_memory_gb > 0 and hw_info.gpu_total_memory_gb <= 4:
        candidate_input_sizes = [128, 192]
        candidate_models = ['resnet18', 'standard', 'large']
    elif hw_info.gpu_total_memory_gb > 6:
        candidate_models = ['resnet50', 'resnet34', 'large', 'standard']
        candidate_input_sizes = [256, 192]
    elif hw_info.gpu_total_memory_gb == 0:
        candidate_models = ['efficientnet_b0', 'standard', 'small']
        candidate_input_sizes = [128]
    else:
        candidate_models = ['resnet34', 'resnet18', 'large', 'standard']
        candidate_input_sizes = [256, 192, 128]
    
    batch_results = {}
    
    for model_name in candidate_models:
        for input_size in candidate_input_sizes:
            if log_fn:
                log_fn(f"  探测 {model_name} @ {input_size}x{input_size}...")
            
            max_batch, results = find_max_batch_size(
                model_name, input_size, device, num_classes,
                start_batch=8, max_batch=256, use_amp=use_amp, log_fn=None,
                total_gpu_memory_gb=hw_info.gpu_total_memory_gb
            )
            
            key = (model_name, input_size)
            batch_results[key] = (max_batch, results)
            
            if max_batch >= 16:
                speed = 0
                for r in results:
                    if r.success and r.batch_size == max_batch:
                        speed = r.samples_per_second
                        break
                
                model_priority = {
                    'resnet50': 10, 'resnet34': 9, 'resnet18': 8,
                    'efficientnet_b2': 7, 'efficientnet_b1': 6, 'efficientnet_b0': 5,
                    'large': 4, 'standard': 3, 'mobilenet': 2, 'small': 1, 'tiny': 0
                }
                current_priority = model_priority.get(best_model, 0)
                new_priority = model_priority.get(model_name, 0)
                
                is_better = False
                if new_priority > current_priority:
                    is_better = True
                elif new_priority == current_priority and input_size > best_input_size:
                    is_better = True
                elif speed > 0 and best_batch_size > 0:
                    current_speed = 0
                    for r in batch_results.get((best_model, best_input_size), (0, []))[1]:
                        if r.success and r.batch_size == best_batch_size:
                            current_speed = r.samples_per_second
                            break
                    if speed > current_speed * 1.2:
                        is_better = True
                
                if is_better or best_batch_size < 16:
                    best_model = model_name
                    best_input_size = input_size
                    best_batch_size = max_batch
    
    if progress_fn:
        progress_fn(2, 3)
    
    if log_fn:
        log_fn("")
        log_fn("[3/3] 数据加载性能测试...")
    
    optimal_workers = 0
    worker_results = {0: 0}
    
    if data_dir and os.path.isdir(data_dir):
        optimal_workers, worker_results = find_optimal_num_workers(
            data_dir, best_input_size, best_batch_size, max_workers=4, log_fn=log_fn
        )
    else:
        if log_fn:
            log_fn("  跳过（未提供数据集目录）")
        if platform.system() == 'Windows':
            optimal_workers = 0
        else:
            optimal_workers = 2
    
    if progress_fn:
        progress_fn(3, 3)
    
    TARGET_TIME_MIN = 2.0
    MAX_TIME_MIN = 3.0
    
    def calculate_epoch_time(model_name, input_size, batch_size):
        key = (model_name, input_size)
        if key not in batch_results:
            return float('inf')
        
        results_list = batch_results[key][1]
        time_per_batch_ms = 100
        for r in results_list:
            if r.success and r.batch_size == batch_size:
                time_per_batch_ms = r.time_per_batch_ms
                break
        
        batches = total_images // batch_size
        train_time = batches * time_per_batch_ms / 1000 / 60
        
        overhead_factor = 1.3
        return train_time * overhead_factor
    
    current_time = calculate_epoch_time(best_model, best_input_size, best_batch_size)
    
    if log_fn:
        log_fn(f"  初选配置: {best_model}@{best_input_size}, 批量{best_batch_size}, 预估{current_time:.1f}分钟/轮")
    
    if current_time > MAX_TIME_MIN:
        if log_fn:
            log_fn(f"  [优化] 预估时间 {current_time:.1f}分钟 > 目标 {MAX_TIME_MIN}分钟，正在寻找更快配置...")
        
        all_configs = []
        for (model_name, input_size), (max_batch, results_list) in batch_results.items():
            if max_batch >= 8:
                for r in results_list:
                    if r.success:
                        epoch_time = calculate_epoch_time(model_name, input_size, r.batch_size)
                        model_priority = {
                            'resnet50': 10, 'resnet34': 9, 'resnet18': 8,
                            'efficientnet_b2': 7, 'efficientnet_b1': 6, 'efficientnet_b0': 5,
                            'large': 4, 'standard': 3, 'mobilenet': 2, 'small': 1, 'tiny': 0
                        }
                        score = model_priority.get(model_name, 0) * 1000 + input_size * 10 + r.batch_size / 10
                        all_configs.append({
                            'model': model_name,
                            'input_size': input_size,
                            'batch_size': r.batch_size,
                            'time': epoch_time,
                            'score': score,
                        })
        
        if all_configs:
            all_configs.sort(key=lambda x: (-x['score'], x['time']))
            
            best_config = None
            for cfg in all_configs:
                if cfg['time'] <= TARGET_TIME_MIN:
                    best_config = cfg
                    break
            
            if best_config is None:
                for cfg in all_configs:
                    if cfg['time'] <= MAX_TIME_MIN:
                        best_config = cfg
                        break
            
            if best_config is None:
                best_config = all_configs[0]
            
            best_model = best_config['model']
            best_input_size = best_config['input_size']
            best_batch_size = best_config['batch_size']
            current_time = best_config['time']
            if log_fn:
                if current_time <= TARGET_TIME_MIN:
                    log_fn(f"  [优化] 选择: {best_model}@{best_input_size}, 批量{best_batch_size}, 预估{current_time:.1f}分钟 ✓")
                elif current_time <= MAX_TIME_MIN:
                    log_fn(f"  [优化] 选择: {best_model}@{best_input_size}, 批量{best_batch_size}, 预估{current_time:.1f}分钟 ○")
                else:
                    log_fn(f"  [优化] 最快配置: {best_model}@{best_input_size}, 批量{best_batch_size}, 预估{current_time:.1f}分钟 ⚠")
    
    time_per_epoch_min = calculate_epoch_time(best_model, best_input_size, best_batch_size)
    
    confidence = "高"
    if time_per_epoch_min > MAX_TIME_MIN:
        confidence = "低"
    elif time_per_epoch_min > TARGET_TIME_MIN:
        confidence = "中"
    elif hw_info.gpu_total_memory_gb > 0 and hw_info.gpu_total_memory_gb <= 4 and best_batch_size < 16:
        confidence = "中"
    if best_batch_size < 8:
        confidence = "低"
    
    accum_steps = 1
    if best_batch_size < 32:
        accum_steps = min(4, max(1, 32 // best_batch_size))
    
    optimal_config = OptimalConfig(
        model_name=best_model,
        input_size=best_input_size,
        batch_size=best_batch_size,
        num_workers=optimal_workers,
        use_amp=use_amp,
        accum_steps=accum_steps,
        estimated_time_per_epoch_min=time_per_epoch_min,
        confidence=confidence,
    )
    
    if log_fn:
        log_fn("")
        log_fn("=" * 60)
        log_fn("[检测结果] 最优配置:")
        log_fn(f"  模型类型: {optimal_config.model_name}")
        log_fn(f"  输入尺寸: {optimal_config.input_size}")
        log_fn(f"  批量大小: {optimal_config.batch_size}")
        log_fn(f"  数据加载线程: {optimal_config.num_workers}")
        log_fn(f"  混合精度: {'启用' if optimal_config.use_amp else '禁用'}")
        log_fn(f"  梯度累积: {optimal_config.accum_steps}步")
        log_fn(f"  预计每轮时间: {optimal_config.estimated_time_per_epoch_min:.1f}分钟")
        if optimal_config.estimated_time_per_epoch_min <= TARGET_TIME_MIN:
            log_fn(f"  ✓ 达到目标时间 (≤{TARGET_TIME_MIN}分钟)")
        elif optimal_config.estimated_time_per_epoch_min <= MAX_TIME_MIN:
            log_fn(f"  ○ 接近目标时间 (≤{MAX_TIME_MIN}分钟)")
        else:
            log_fn(f"  ⚠ 超出目标时间 (>{MAX_TIME_MIN}分钟)")
        log_fn(f"  置信度: {optimal_config.confidence}")
        log_fn("=" * 60)
    
    return hw_info, optimal_config


def format_hardware_report(hw_info: HardwareInfo) -> str:
    lines = [
        "╔══════════════════════════════════════════════════════════╗",
        "║                    硬件检测报告                          ║",
        "╠══════════════════════════════════════════════════════════╣",
        f"║ CPU: {hw_info.cpu_name:<48}║",
        f"║ 核心数: {hw_info.cpu_cores}核 / {hw_info.cpu_threads}线程{' ' * (38 - len(str(hw_info.cpu_cores)) - len(str(hw_info.cpu_threads)))}║",
        f"║ 内存: {hw_info.ram_total_gb:.1f}GB (可用: {hw_info.ram_available_gb:.1f}GB){' ' * (28 - len(f'{hw_info.ram_total_gb:.1f}') - len(f'{hw_info.ram_available_gb:.1f}'))}║",
        "╠══════════════════════════════════════════════════════════╣",
        f"║ GPU: {hw_info.gpu_name:<49}║",
    ]
    
    if hw_info.gpu_total_memory_gb > 0:
        lines.append(f"║ 显存: {hw_info.gpu_total_memory_gb:.1f}GB (可用: {hw_info.gpu_available_memory_gb:.1f}GB){' ' * (28 - len(f'{hw_info.gpu_total_memory_gb:.1f}') - len(f'{hw_info.gpu_available_memory_gb:.1f}'))}║")
        lines.append(f"║ 计算能力: {hw_info.gpu_compute_capability:<44}║")
    else:
        lines.append("║ 显存: 无GPU                                              ║")
    
    lines.extend([
        "╠══════════════════════════════════════════════════════════╣",
        f"║ 平台: {hw_info.platform_info:<48}║",
        "╚══════════════════════════════════════════════════════════╝",
    ])
    
    return '\n'.join(lines)


def format_config_report(config: OptimalConfig) -> str:
    lines = [
        "╔══════════════════════════════════════════════════════════╗",
        "║                    推荐训练配置                          ║",
        "╠══════════════════════════════════════════════════════════╣",
        f"║ 模型类型:     {config.model_name:<42}║",
        f"║ 输入尺寸:     {config.input_size}x{config.input_size}{' ' * (40 - len(str(config.input_size)) * 2)}║",
        f"║ 批量大小:     {config.batch_size:<42}║",
        f"║ 数据加载线程: {config.num_workers:<42}║",
        f"║ 混合精度:     {'启用' if config.use_amp else '禁用'}{' ' * (42 - (2 if config.use_amp else 2))}║",
        f"║ 梯度累积:     {config.accum_steps}步{' ' * (41 - len(str(config.accum_steps)))}║",
        "╠══════════════════════════════════════════════════════════╣",
        f"║ 预计每轮时间: {config.estimated_time_per_epoch_min:.1f} 分钟{' ' * (35 - len(f'{config.estimated_time_per_epoch_min:.1f}'))}║",
        f"║ 置信度:       {config.confidence:<42}║",
        "╚══════════════════════════════════════════════════════════╝",
    ]
    
    return '\n'.join(lines)
