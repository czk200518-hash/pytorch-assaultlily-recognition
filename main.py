import sys
import os
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def cmd_train(args):
    from core.train import train_model

    train_model(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        model_name=args.model,
        input_size=args.input_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        val_split=args.val_split,
        num_workers=args.num_workers,
        resume=args.resume,
    )


def cmd_predict(args):
    from core.predict import load_model, predict_image, predict_folder

    device = None
    model, class_names, device = load_model(args.model_path, device)

    print(f'已加载模型，类别数量: {len(class_names)}')
    print(f'类别名称: {class_names}')

    if args.folder:
        results = predict_folder(
            model, args.image, class_names, device,
            input_size=args.input_size,
        )
        print(f'\n{"="*60}')
        print(f'文件夹预测结果 ({args.image}):')
        print(f'{"="*60}')
        for filename, result in results.items():
            print(f'\n  [{filename}]')
            print(f'    预测: {result["predicted_class"]}')
            print(f'    置信度: {result["confidence"]:.4f} ({result["confidence"]*100:.1f}%)')
            print(f'    Top-3:')
            for name, prob in result['top3']:
                print(f'      - {name}: {prob*100:.1f}%')
    else:
        result = predict_image(
            model, args.image, class_names, device,
            input_size=args.input_size,
        )
        print(f'\n{"="*60}')
        print(f'单张图片预测结果:')
        print(f'{"="*60}')
        print(f'  预测: {result["predicted_class"]}')
        print(f'  置信度: {result["confidence"]:.4f} ({result["confidence"]*100:.1f}%)')
        print(f'  Top-3:')
        for name, prob in result['top3']:
            print(f'    - {name}: {prob*100:.1f}%')


def cmd_info(args):
    from core.predict import load_model

    try:
        model, class_names, device = load_model(args.model_path)
        print(f'\n{"="*60}')
        print(f'模型信息:')
        print(f'{"="*60}')
        print(f'  模型文件: {args.model_path}')
        print(f'  设备: {device}')
        print(f'  类别数量: {len(class_names)}')
        print(f'  类别列表:')
        for i, name in enumerate(class_names):
            print(f'    [{i}] {name}')
    except Exception as e:
        print(f'无法加载模型: {e}')
        sys.exit(1)


def cmd_test(args):
    from core.predict import load_model, predict_image
    from core.dataset_loader import load_test_dataset
    import torch

    device = None
    model, class_names, device = load_model(args.model_path, device)

    if not args.data_dir:
        print('错误: 测试模式需要指定 --data_dir 参数')
        sys.exit(1)

    test_loader, _, _ = load_test_dataset(
        args.data_dir, input_size=args.input_size,
        batch_size=1,
    )

    correct = 0
    total = 0

    print(f'\n评估模型在测试集上的表现...\n')

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    acc = 100. * correct / total
    print(f'测试准确率: {acc:.2f}%')
    print(f'正确: {correct}/{total}')


def cmd_detect(args):
    from utils import face_detector as fd

    min_size = (args.min_size, args.min_size)
    export_mode = 'merged' if args.merged else 'separate'
    strict_filter = args.strict

    if os.path.isfile(args.image):
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)

            base_name = Path(args.image).stem

            if args.draw_box:
                if export_mode == 'merged':
                    annotated_path = os.path.join(
                        args.output_dir,
                        f'标注_{base_name}.png'
                    )
                else:
                    annotated_path = os.path.join(
                        args.output_dir, base_name,
                        f'标注_{base_name}.png'
                    )
                _, count = fd.draw_boxes(
                    args.image, annotated_path,
                    min_size=min_size,
                    min_neighbors_low=args.neighbors_low,
                    min_neighbors_high=args.neighbors_high,
                    intensity=args.intensity,
                    strict_filter=strict_filter,
                )
                print(f'标注图已保存 ({count} 个人脸): {annotated_path}')

            results = fd.crop_faces(
                args.image, args.output_dir,
                min_size=min_size,
                min_neighbors_low=args.neighbors_low,
                min_neighbors_high=args.neighbors_high,
                padding=args.padding,
                intensity=args.intensity,
                export_mode=export_mode,
                strict_filter=strict_filter,
            )
            sub_dir = args.output_dir if export_mode == 'merged' else os.path.join(args.output_dir, base_name)
            print(f'\n检测到 {len(results)} 个人脸')
            print(f'输出目录: {sub_dir}')
            for r in results:
                print(f'  人脸 #{r["index"]}: 位置({r["x"]},{r["y"]}) '
                      f'尺寸({r["width"]}×{r["height"]}) → {os.path.basename(r["save_path"])}')

        else:
            faces = fd.detect_faces(
                args.image,
                min_size=min_size,
                min_neighbors_low=args.neighbors_low,
                min_neighbors_high=args.neighbors_high,
                intensity=args.intensity,
                strict_filter=strict_filter,
            )
            print(f'\n检测到 {len(faces)} 个人脸:')
            for i, (x, y, w, h) in enumerate(faces):
                print(f'  人脸 #{i+1}: 位置({x},{y}) 尺寸({w}×{h})')

    elif os.path.isdir(args.image):
        output_dir = args.output_dir or os.path.join(
            os.path.dirname(args.image), 'detected_faces'
        )
        results, total = fd.batch_detect_folder(
            args.image, output_dir,
            min_size=min_size,
            min_neighbors_low=args.neighbors_low,
            min_neighbors_high=args.neighbors_high,
            padding=args.padding,
            intensity=args.intensity,
            export_mode=export_mode,
            strict_filter=strict_filter,
        )
        print(f'\n批量检测完成! 共处理 {len(results)} 张图片，检测到 {total} 个人脸')
        print(f'输出目录: {output_dir}')
    else:
        print(f'错误: 无效的路径 - {args.image}')
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='动漫人脸识别工具 - 基于 PyTorch CNN',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例用法:
  训练模型:
    python main.py train --data_dir ./dataset/characters --model_dir ./models --epochs 50

  预测单张图片:
    python main.py predict --model_path ./models/anime_face_xxx.pth --image ./test.jpg

  预测文件夹:
    python main.py predict --model_path ./models/anime_face_xxx.pth --image ./test_images/ --folder

  查看模型信息:
    python main.py info --model_path ./models/anime_face_xxx.pth

  测试准确率:
    python main.py test --model_path ./models/anime_face_xxx.pth --data_dir ./dataset/test/

  人脸检测（仅检测，不裁剪）:
    python main.py detect --image ./screenshot.jpg

  人脸检测并裁剪保存（彻底扫描）:
    python main.py detect --image ./screenshot.jpg --output_dir ./faces/ --draw_box --intensity thorough

  合并导出（所有人脸放在同一文件夹）:
    python main.py detect --image ./images_folder/ --output_dir ./all_faces/ --merged

  快速批量检测文件夹:
    python main.py detect --image ./images_folder/ --output_dir ./all_faces/ --intensity fast

数据集目录结构:
  dataset/
    ├── 鸣人/
    │   ├── 001.jpg
    │   └── 002.jpg
    ├── 佐助/
    │   └── 001.jpg
    └── ...
        ''',
    )

    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('--data_dir', type=str, default='./dataset',
                              help='数据集目录路径 (默认: ./dataset)')
    train_parser.add_argument('--model_dir', type=str, default='./models',
                              help='模型保存目录 (默认: ./models)')
    train_parser.add_argument('--model', type=str, default='standard',
                              choices=['standard', 'small'],
                              help='模型类型: standard(大模型) / small(轻量) (默认: standard)')
    train_parser.add_argument('--input_size', type=int, default=128,
                              help='输入图片尺寸 (默认: 128)')
    train_parser.add_argument('--batch_size', type=int, default=32,
                              help='批量大小 (默认: 32)')
    train_parser.add_argument('--epochs', type=int, default=50,
                              help='训练轮数 (默认: 50)')
    train_parser.add_argument('--lr', type=float, default=0.001,
                              help='学习率 (默认: 0.001)')
    train_parser.add_argument('--val_split', type=float, default=0.2,
                              help='验证集比例 (默认: 0.2)')
    train_parser.add_argument('--num_workers', type=int, default=0,
                              help='DataLoader 工作进程数 (默认: 0)')
    train_parser.add_argument('--resume', type=str, default=None,
                              help='从检查点恢复训练')

    predict_parser = subparsers.add_parser('predict', help='预测图片')
    predict_parser.add_argument('--model_path', type=str, required=True,
                                help='训练好的模型文件路径')
    predict_parser.add_argument('--image', type=str, required=True,
                                help='要预测的图片路径或文件夹路径')
    predict_parser.add_argument('--folder', action='store_true',
                                help='对文件夹内所有图片进行预测')
    predict_parser.add_argument('--input_size', type=int, default=128,
                                help='输入图片尺寸，需与训练时一致 (默认: 128)')

    info_parser = subparsers.add_parser('info', help='查看模型信息')
    info_parser.add_argument('--model_path', type=str, required=True,
                             help='模型文件路径')

    test_parser = subparsers.add_parser('test', help='在测试集上评估模型')
    test_parser.add_argument('--model_path', type=str, required=True,
                             help='训练好的模型文件路径')
    test_parser.add_argument('--data_dir', type=str, default=None,
                             help='测试数据集目录路径')
    test_parser.add_argument('--input_size', type=int, default=128,
                             help='输入图片尺寸，需与训练时一致 (默认: 128)')

    detect_parser = subparsers.add_parser('detect', help='检测动漫人脸并裁剪')
    detect_parser.add_argument('--image', type=str, required=True,
                               help='要检测的图片路径或文件夹路径')
    detect_parser.add_argument('--output_dir', type=str, default=None,
                               help='裁剪输出目录（不指定则仅检测）')
    detect_parser.add_argument('--draw_box', action='store_true',
                               help='同时保存标注了人脸框的图片')
    detect_parser.add_argument('--min_size', type=int, default=32,
                               help='最小人脸尺寸 (默认: 32)')
    detect_parser.add_argument('--neighbors_low', type=int, default=3,
                               help='邻居下限，越低召回越高 (默认: 3)')
    detect_parser.add_argument('--neighbors_high', type=int, default=7,
                               help='邻居上限，越高误检越少 (默认: 7)')
    detect_parser.add_argument('--padding', type=int, default=10,
                               help='裁剪外扩像素数 (默认: 10)')
    detect_parser.add_argument('--intensity', type=str, default='standard',
                               choices=['fast', 'standard', 'thorough'],
                               help='检测强度: fast|standard|thorough (默认: standard)')
    detect_parser.add_argument('--merged', action='store_true',
                               help='合并导出: 所有人脸直接保存在输出目录，不创建子文件夹')
    detect_parser.add_argument('--strict', action='store_true',
                               help='严格过滤: 排除衣服/腿部等非人脸误检')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    command_map = {
        'train': cmd_train,
        'predict': cmd_predict,
        'info': cmd_info,
        'test': cmd_test,
        'detect': cmd_detect,
    }

    command_map[args.command](args)


if __name__ == '__main__':
    main()
