#!/usr/bin/env python3
"""
YOLO11 Training Script for COCO128 Dataset
Usage: python train_yolo11_coco128.py
"""

import os
import argparse
from pathlib import Path
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description='Train YOLO11 on COCO128 dataset')
    parser.add_argument('--model', type=str, default='models/yolo11n.yaml',
                       help='YOLO model to use (yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt, or custom yaml)')
    parser.add_argument('--data', type=str, default='/home/shyue/codebase/YOLO/datasets/HIT_UAV_YOLO/data.yaml',
                       help='Dataset to use (coco128, or custom yaml)')
    parser.add_argument('--epochs', type=int, default=12,
                       help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, 
                       help='Image size for training')
    parser.add_argument('--name', type=str, default='yolo11_swin_coco128', 
                       help='Experiment name')
    parser.add_argument('--project', type=str, default='runs/detect', 
                       help='Project directory to save results')
    parser.add_argument('--device', type=str, default='0',
                       help='Device to train on (e.g., 0, cpu)')
    parser.add_argument('--workers', type=int, default=8, 
                       help='Number of dataloader workers')
    parser.add_argument('--save_period', type=int, default=5, 
                       help='Save checkpoint every x epochs')
    
    args = parser.parse_args()
    
    # Set working directory
    script_dir = Path(__file__).parent.absolute()
    os.chdir(script_dir)
    print(f"Working directory: {script_dir}")
    
    # Check if dataset exists
    dataset_path = "/home/shyue/codebase/YOLO/datasets/HIT_UAV_YOLO/data.yaml"
    if not Path(dataset_path).exists():
        print(f"Error: Dataset configuration not found at {dataset_path}")
        print("Please ensure the COCO128 dataset is downloaded and configured properly.")
        return
    
    # Initialize model
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)
    
    # Configure training parameters
    train_args = {
        'data': str(dataset_path),
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch,
        'name': args.name,
        'project': args.project,
        'save_period': args.save_period,
        'workers': args.workers,
        'plots': True,
        'val': True,
        'save': True,
        'exist_ok': True,
    }
    
    # Add device if specified
    if args.device:
        train_args['device'] = args.device
    
    print(f"Training configuration:")
    for key, value in train_args.items():
        print(f"  {key}: {value}")
    
    print("\nStarting training...")
    
    # Train the model
    try:
        results = model.train(**train_args)
        
        print(f"\nTraining completed successfully!")
        print(f"Results saved to: {results.save_dir}")
        print(f"Best model: {results.save_dir}/weights/best.pt")
        print(f"Last model: {results.save_dir}/weights/last.pt")
        
        # Print final metrics
        if hasattr(results, 'metrics'):
            print(f"\nFinal metrics:")
            metrics = results.metrics
            if hasattr(metrics, 'box'):
                print(f"  mAP@50: {metrics.box.map50:.4f}")
                print(f"  mAP@50-95: {metrics.box.map:.4f}")
                print(f"  Precision: {metrics.box.mp:.4f}")
                print(f"  Recall: {metrics.box.mr:.4f}")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise

if __name__ == "__main__":
    main()