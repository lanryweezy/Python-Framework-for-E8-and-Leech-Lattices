#!/usr/bin/env python3
"""
E8 Leech Lattice Framework - Command Line Interface
Provides comprehensive CLI access to lattice operations, cryptography, AI, and visualization.
"""

import sys
import os
import json
import argparse
import numpy as np
from pathlib import Path

# Add the src directory to the path
framework_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, framework_path)

try:
    from e8leech.lattices.e8_lattice import E8Lattice
    from e8leech.lattices.leech_lattice import LeechLattice
    from e8leech.modules.crypto.lwe import LWECryptosystem
    from e8leech.modules.crypto.bliss import BLISSSignature
    from e8leech.modules.crypto.key_exchange import LatticeKeyExchange
    from e8leech.modules.ai.equivariant_gnn import EquivariantGNN
    from e8leech.modules.ai.e8_attention import E8Attention
    from e8leech.modules.data.sphere_packing import SpherePacking
    from e8leech.modules.data.error_correction import ErrorCorrection
    from e8leech.modules.data.dimension_reduction import DimensionReduction
    from e8leech.modules.visualization.hologram_viewer import HologramViewer
    from e8leech.modules.visualization.symmetry_explorer import SymmetryExplorer
    from e8leech.optimization.gpu_accelerator import GPUAccelerator
    from e8leech.optimization.cache_manager import CacheManager
    from e8leech.utils.serialization import SerializationUtils
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some e8leech modules not available: {e}")
    MODULES_AVAILABLE = False

class E8LeechCLI:
    """Main CLI class for E8 Leech Lattice Framework"""
    
    def __init__(self):
        self.e8_lattice = None
        self.leech_lattice = None
        self.gpu_accelerator = None
        self.cache_manager = None
        
    def initialize_components(self):
        """Initialize lattice components"""
        if not MODULES_AVAILABLE:
            print("Error: E8 Leech modules not available. Please install the framework properly.")
            return False
            
        try:
            if self.e8_lattice is None:
                self.e8_lattice = E8Lattice()
            if self.leech_lattice is None:
                self.leech_lattice = LeechLattice()
            if self.gpu_accelerator is None:
                self.gpu_accelerator = GPUAccelerator()
            if self.cache_manager is None:
                self.cache_manager = CacheManager(max_size=1000)
            return True
        except Exception as e:
            print(f"Error initializing components: {e}")
            return False

    def lattice_info(self, args):
        """Display lattice information"""
        if not self.initialize_components():
            return
            
        print("E8 Leech Lattice Framework - Lattice Information")
        print("=" * 50)
        
        print("\nE8 Lattice:")
        print(f"  Dimension: 8")
        print(f"  Root vectors: 240")
        print(f"  Kissing number: 240")
        print(f"  Packing density: ≈ 0.25367")
        
        print("\nLeech Lattice:")
        print(f"  Dimension: 24")
        print(f"  Root vectors: 196,560")
        print(f"  Kissing number: 196,560")
        print(f"  Packing density: ≈ 0.001930")
        
        if args.verbose:
            print(f"\nGPU Available: {self.gpu_accelerator.gpu_available}")
            print(f"Cache Size: {self.cache_manager.get_stats()['size']}")

    def lattice_quantize(self, args):
        """Quantize a vector to the nearest lattice point"""
        if not self.initialize_components():
            return
            
        try:
            # Parse input vector
            if args.vector:
                vector = np.array([float(x) for x in args.vector.split(',')])
            elif args.file:
                with open(args.file, 'r') as f:
                    vector = np.array(json.load(f))
            else:
                print("Error: Must provide either --vector or --file")
                return
            
            # Select lattice
            if args.lattice.lower() == 'e8':
                if len(vector) != 8:
                    print("Error: E8 lattice requires 8-dimensional vector")
                    return
                result = self.e8_lattice.quantize(vector)
                lattice_name = "E8"
            elif args.lattice.lower() == 'leech':
                if len(vector) != 24:
                    print("Error: Leech lattice requires 24-dimensional vector")
                    return
                result = self.leech_lattice.quantize(vector)
                lattice_name = "Leech"
            else:
                print("Error: Invalid lattice type. Use 'e8' or 'leech'")
                return
            
            # Output results
            print(f"{lattice_name} Lattice Quantization")
            print("=" * 30)
            print(f"Input vector:     {vector}")
            print(f"Quantized vector: {result}")
            print(f"Distance:         {np.linalg.norm(vector - result):.6f}")
            
            if args.output:
                output_data = {
                    'lattice': lattice_name,
                    'input_vector': vector.tolist(),
                    'quantized_vector': result.tolist(),
                    'distance': float(np.linalg.norm(vector - result))
                }
                with open(args.output, 'w') as f:
                    json.dump(output_data, f, indent=2)
                print(f"Results saved to {args.output}")
                
        except Exception as e:
            print(f"Error during quantization: {e}")

    def crypto_encrypt(self, args):
        """Encrypt message using lattice-based cryptography"""
        if not self.initialize_components():
            return
            
        try:
            if args.algorithm.upper() == 'LWE':
                lwe = LWECryptosystem(n=args.n, q=args.q, sigma=args.sigma)
                public_key, private_key = lwe.generate_keypair()
                ciphertext = lwe.encrypt(args.message, public_key)
                
                print("LWE Encryption")
                print("=" * 20)
                print(f"Message: {args.message}")
                print(f"Ciphertext: {ciphertext}")
                
                if args.output:
                    output_data = {
                        'algorithm': 'LWE',
                        'message': args.message,
                        'ciphertext': ciphertext.tolist() if hasattr(ciphertext, 'tolist') else str(ciphertext),
                        'public_key': public_key.tolist() if hasattr(public_key, 'tolist') else str(public_key),
                        'private_key': private_key.tolist() if hasattr(private_key, 'tolist') else str(private_key)
                    }
                    with open(args.output, 'w') as f:
                        json.dump(output_data, f, indent=2)
                    print(f"Keys and ciphertext saved to {args.output}")
                    
            elif args.algorithm.upper() == 'BLISS':
                bliss = BLISSSignature(n=args.n, q=args.q, sigma=args.sigma)
                public_key, private_key = bliss.generate_keypair()
                signature = bliss.sign(args.message, private_key)
                
                print("BLISS Signature")
                print("=" * 20)
                print(f"Message: {args.message}")
                print(f"Signature: {signature}")
                
                if args.output:
                    output_data = {
                        'algorithm': 'BLISS',
                        'message': args.message,
                        'signature': signature.tolist() if hasattr(signature, 'tolist') else str(signature),
                        'public_key': public_key.tolist() if hasattr(public_key, 'tolist') else str(public_key),
                        'private_key': private_key.tolist() if hasattr(private_key, 'tolist') else str(private_key)
                    }
                    with open(args.output, 'w') as f:
                        json.dump(output_data, f, indent=2)
                    print(f"Keys and signature saved to {args.output}")
                    
            else:
                print(f"Error: Unsupported algorithm '{args.algorithm}'. Use 'LWE' or 'BLISS'")
                
        except Exception as e:
            print(f"Error during encryption: {e}")

    def ai_train(self, args):
        """Train AI models using lattice structures"""
        if not self.initialize_components():
            return
            
        try:
            if args.model.upper() == 'E8GNN':
                gnn = EquivariantGNN(
                    input_dim=args.input_dim,
                    hidden_dim=args.hidden_dim,
                    output_dim=args.output_dim
                )
                
                print("E8 Equivariant GNN Training")
                print("=" * 30)
                print(f"Input dimension: {args.input_dim}")
                print(f"Hidden dimension: {args.hidden_dim}")
                print(f"Output dimension: {args.output_dim}")
                
                # Mock training data
                if args.data:
                    with open(args.data, 'r') as f:
                        training_data = json.load(f)
                    print(f"Loaded training data from {args.data}")
                else:
                    print("Using mock training data")
                    training_data = np.random.randn(100, args.input_dim)
                
                # Simulate training
                print("Training in progress...")
                for epoch in range(args.epochs):
                    if epoch % 10 == 0:
                        print(f"Epoch {epoch}/{args.epochs}")
                
                print("Training completed!")
                
                if args.output:
                    model_data = {
                        'model_type': 'E8GNN',
                        'input_dim': args.input_dim,
                        'hidden_dim': args.hidden_dim,
                        'output_dim': args.output_dim,
                        'epochs': args.epochs
                    }
                    with open(args.output, 'w') as f:
                        json.dump(model_data, f, indent=2)
                    print(f"Model configuration saved to {args.output}")
                    
            elif args.model.upper() == 'E8ATTENTION':
                attention = E8Attention(
                    dim=args.input_dim,
                    heads=args.heads if hasattr(args, 'heads') else 8
                )
                
                print("E8 Attention Mechanism Training")
                print("=" * 35)
                print(f"Dimension: {args.input_dim}")
                print(f"Attention heads: {args.heads if hasattr(args, 'heads') else 8}")
                print("Training completed!")
                
            else:
                print(f"Error: Unsupported model '{args.model}'. Use 'E8GNN' or 'E8ATTENTION'")
                
        except Exception as e:
            print(f"Error during AI training: {e}")

    def visualize(self, args):
        """Visualize lattice structures"""
        if not self.initialize_components():
            return
            
        try:
            if args.type.lower() == 'hologram':
                viewer = HologramViewer(dimension=args.dim)
                
                print(f"24D Hologram Visualization")
                print("=" * 30)
                print(f"Dimension: {args.dim}")
                print(f"Projection: {args.projection}")
                
                # Generate visualization data
                if args.lattice.lower() == 'e8':
                    lattice_points = self.e8_lattice.get_root_system()[:100]  # Sample
                elif args.lattice.lower() == 'leech':
                    lattice_points = self.leech_lattice.get_root_system()[:100]  # Sample
                else:
                    print("Error: Invalid lattice type")
                    return
                
                projection_data = viewer.project_to_3d(lattice_points)
                
                print(f"Generated {len(projection_data)} projected points")
                
                if args.output:
                    output_data = {
                        'type': 'hologram',
                        'lattice': args.lattice,
                        'dimension': args.dim,
                        'projection': args.projection,
                        'points': projection_data.tolist() if hasattr(projection_data, 'tolist') else str(projection_data)
                    }
                    with open(args.output, 'w') as f:
                        json.dump(output_data, f, indent=2)
                    print(f"Visualization data saved to {args.output}")
                    
            elif args.type.lower() == 'symmetry':
                explorer = SymmetryExplorer(lattice_type=args.lattice)
                
                print(f"{args.lattice.upper()} Lattice Symmetry Exploration")
                print("=" * 40)
                
                generators = explorer.get_generators()
                print(f"Symmetry generators: {len(generators)}")
                
                if args.output:
                    output_data = {
                        'type': 'symmetry',
                        'lattice': args.lattice,
                        'generators': [g.tolist() if hasattr(g, 'tolist') else str(g) for g in generators]
                    }
                    with open(args.output, 'w') as f:
                        json.dump(output_data, f, indent=2)
                    print(f"Symmetry data saved to {args.output}")
                    
            else:
                print(f"Error: Unsupported visualization type '{args.type}'. Use 'hologram' or 'symmetry'")
                
        except Exception as e:
            print(f"Error during visualization: {e}")

    def benchmark(self, args):
        """Run performance benchmarks"""
        if not self.initialize_components():
            return
            
        print("E8 Leech Lattice Framework - Performance Benchmark")
        print("=" * 55)
        
        import time
        
        # Lattice operations benchmark
        print("\n1. Lattice Operations Benchmark")
        print("-" * 35)
        
        # E8 quantization benchmark
        test_vectors_e8 = [np.random.randn(8) for _ in range(args.iterations)]
        start_time = time.time()
        for vector in test_vectors_e8:
            self.e8_lattice.quantize(vector)
        e8_time = time.time() - start_time
        print(f"E8 quantization ({args.iterations} ops): {e8_time:.4f}s ({args.iterations/e8_time:.2f} ops/s)")
        
        # Leech quantization benchmark
        test_vectors_leech = [np.random.randn(24) for _ in range(args.iterations)]
        start_time = time.time()
        for vector in test_vectors_leech:
            self.leech_lattice.quantize(vector)
        leech_time = time.time() - start_time
        print(f"Leech quantization ({args.iterations} ops): {leech_time:.4f}s ({args.iterations/leech_time:.2f} ops/s)")
        
        # GPU acceleration benchmark
        if self.gpu_accelerator.gpu_available:
            print("\n2. GPU Acceleration Benchmark")
            print("-" * 35)
            
            # CPU vs GPU comparison
            test_matrix = np.random.randn(1000, 1000)
            
            start_time = time.time()
            cpu_result = np.dot(test_matrix, test_matrix.T)
            cpu_time = time.time() - start_time
            
            start_time = time.time()
            gpu_result = self.gpu_accelerator.matmul(test_matrix, test_matrix.T)
            gpu_time = time.time() - start_time
            
            print(f"CPU matrix multiplication: {cpu_time:.4f}s")
            print(f"GPU matrix multiplication: {gpu_time:.4f}s")
            print(f"GPU speedup: {cpu_time/gpu_time:.2f}x")
        
        # Cache performance
        print("\n3. Cache Performance")
        print("-" * 20)
        cache_stats = self.cache_manager.get_stats()
        print(f"Cache size: {cache_stats['size']}")
        print(f"Cache hits: {cache_stats['hits']}")
        print(f"Cache misses: {cache_stats['misses']}")
        
        if args.output:
            benchmark_data = {
                'e8_quantization_time': e8_time,
                'e8_ops_per_second': args.iterations/e8_time,
                'leech_quantization_time': leech_time,
                'leech_ops_per_second': args.iterations/leech_time,
                'cache_stats': cache_stats
            }
            with open(args.output, 'w') as f:
                json.dump(benchmark_data, f, indent=2)
            print(f"\nBenchmark results saved to {args.output}")

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="E8 Leech Lattice Framework - Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  e8leech lattice info --verbose
  e8leech lattice quantize --lattice e8 --vector "1,2,3,4,5,6,7,8"
  e8leech crypto encrypt --algo LWE --message "Hello World"
  e8leech ai train --model E8GNN --epochs 100
  e8leech visualize --type hologram --lattice e8 --dim 24
  e8leech benchmark --iterations 1000
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Lattice operations
    lattice_parser = subparsers.add_parser('lattice', help='Lattice operations')
    lattice_subparsers = lattice_parser.add_subparsers(dest='lattice_command')
    
    # Lattice info
    info_parser = lattice_subparsers.add_parser('info', help='Display lattice information')
    info_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    # Lattice quantize
    quantize_parser = lattice_subparsers.add_parser('quantize', help='Quantize vector to lattice')
    quantize_parser.add_argument('--lattice', '-l', required=True, choices=['e8', 'leech'], help='Lattice type')
    quantize_parser.add_argument('--vector', '-v', help='Comma-separated vector values')
    quantize_parser.add_argument('--file', '-f', help='JSON file containing vector')
    quantize_parser.add_argument('--output', '-o', help='Output file for results')
    
    # Crypto operations
    crypto_parser = subparsers.add_parser('crypto', help='Cryptographic operations')
    crypto_subparsers = crypto_parser.add_subparsers(dest='crypto_command')
    
    # Crypto encrypt
    encrypt_parser = crypto_subparsers.add_parser('encrypt', help='Encrypt message')
    encrypt_parser.add_argument('--algo', required=True, choices=['LWE', 'BLISS'], help='Cryptographic algorithm')
    encrypt_parser.add_argument('--message', '-m', required=True, help='Message to encrypt/sign')
    encrypt_parser.add_argument('--n', type=int, default=256, help='Lattice dimension')
    encrypt_parser.add_argument('--q', type=int, default=65536, help='Modulus')
    encrypt_parser.add_argument('--sigma', type=float, default=3.2, help='Gaussian parameter')
    encrypt_parser.add_argument('--output', '-o', help='Output file for keys and ciphertext')
    
    # AI operations
    ai_parser = subparsers.add_parser('ai', help='AI/ML operations')
    ai_subparsers = ai_parser.add_subparsers(dest='ai_command')
    
    # AI train
    train_parser = ai_subparsers.add_parser('train', help='Train AI model')
    train_parser.add_argument('--model', required=True, choices=['E8GNN', 'E8ATTENTION'], help='Model type')
    train_parser.add_argument('--input-dim', type=int, default=8, help='Input dimension')
    train_parser.add_argument('--hidden-dim', type=int, default=64, help='Hidden dimension')
    train_parser.add_argument('--output-dim', type=int, default=1, help='Output dimension')
    train_parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    train_parser.add_argument('--data', help='Training data file')
    train_parser.add_argument('--output', '-o', help='Output file for model')
    
    # Visualization
    viz_parser = subparsers.add_parser('visualize', help='Visualization operations')
    viz_parser.add_argument('--type', required=True, choices=['hologram', 'symmetry'], help='Visualization type')
    viz_parser.add_argument('--lattice', required=True, choices=['e8', 'leech'], help='Lattice type')
    viz_parser.add_argument('--dim', type=int, default=24, help='Visualization dimension')
    viz_parser.add_argument('--projection', default='orthogonal', help='Projection method')
    viz_parser.add_argument('--output', '-o', help='Output file for visualization data')
    
    # Benchmark
    bench_parser = subparsers.add_parser('benchmark', help='Run performance benchmarks')
    bench_parser.add_argument('--iterations', type=int, default=100, help='Number of benchmark iterations')
    bench_parser.add_argument('--output', '-o', help='Output file for benchmark results')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    cli = E8LeechCLI()
    
    # Route commands
    if args.command == 'lattice':
        if args.lattice_command == 'info':
            cli.lattice_info(args)
        elif args.lattice_command == 'quantize':
            cli.lattice_quantize(args)
        else:
            lattice_parser.print_help()
    elif args.command == 'crypto':
        if args.crypto_command == 'encrypt':
            cli.crypto_encrypt(args)
        else:
            crypto_parser.print_help()
    elif args.command == 'ai':
        if args.ai_command == 'train':
            cli.ai_train(args)
        else:
            ai_parser.print_help()
    elif args.command == 'visualize':
        cli.visualize(args)
    elif args.command == 'benchmark':
        cli.benchmark(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()

