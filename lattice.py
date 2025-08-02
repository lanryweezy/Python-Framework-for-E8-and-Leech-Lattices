from flask import Blueprint, request, jsonify
import numpy as np
import sys
import os

# Add the main framework to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from e8leech.lattices.e8_lattice import E8Lattice
from e8leech.lattices.leech_lattice import LeechLattice
from e8leech.modules.crypto.lwe import LWE
from e8leech.modules.ai.equivariant_gnn import EquivariantGNN

lattice_bp = Blueprint('lattice', __name__)

# Global instances for reuse
e8_lattice = None
leech_lattice = None
lwe_crypto = None
equivariant_gnn = None

def initialize_lattices():
    global e8_lattice, leech_lattice, lwe_crypto, equivariant_gnn
    if e8_lattice is None:
        e8_lattice = E8Lattice()
    if leech_lattice is None:
        leech_lattice = LeechLattice()
    if lwe_crypto is None:
        lwe_crypto = LWE(dimension=24, modulus=257)
    if equivariant_gnn is None:
        equivariant_gnn = EquivariantGNN(input_dim=24, hidden_dim=64, output_dim=24)

@lattice_bp.route('/e8/info', methods=['GET'])
def e8_info():
    """Get E8 lattice information"""
    initialize_lattices()
    return jsonify({
        'name': 'E8 Lattice',
        'dimension': 8,
        'root_count': 240,
        'kissing_number': 240,
        'packing_density': 0.25367,
        'properties': e8_lattice.get_expected_properties()
    })

@lattice_bp.route('/e8/roots', methods=['GET'])
def e8_roots():
    """Generate E8 root system"""
    initialize_lattices()
    try:
        roots = e8_lattice.generate_root_system()
        return jsonify({
            'root_count': len(roots),
            'roots': roots.tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@lattice_bp.route('/e8/quantize', methods=['POST'])
def e8_quantize():
    """Quantize vectors to E8 lattice points"""
    initialize_lattices()
    try:
        data = request.get_json()
        vectors = np.array(data['vectors'])
        
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        
        quantized = e8_lattice.quantize(vectors)
        return jsonify({
            'quantized_vectors': quantized.tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@lattice_bp.route('/e8/nearest', methods=['POST'])
def e8_nearest():
    """Find nearest E8 lattice point"""
    initialize_lattices()
    try:
        data = request.get_json()
        target = np.array(data['target'])
        algorithm = data.get('algorithm', 'babai')
        
        nearest = e8_lattice.nearest_neighbor(target, algorithm)
        distance = np.linalg.norm(target - nearest)
        
        return jsonify({
            'nearest_point': nearest.tolist(),
            'distance': float(distance)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@lattice_bp.route('/leech/info', methods=['GET'])
def leech_info():
    """Get Leech lattice information"""
    initialize_lattices()
    return jsonify({
        'name': 'Leech Lattice',
        'dimension': 24,
        'kissing_number': 196560,
        'properties': leech_lattice.get_expected_properties()
    })

@lattice_bp.route('/leech/quantize', methods=['POST'])
def leech_quantize():
    """Quantize vectors to Leech lattice points"""
    initialize_lattices()
    try:
        data = request.get_json()
        vectors = np.array(data['vectors'])
        
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        
        quantized = leech_lattice.quantize(vectors)
        return jsonify({
            'quantized_vectors': quantized.tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@lattice_bp.route('/leech/nearest', methods=['POST'])
def leech_nearest():
    """Find nearest Leech lattice point"""
    initialize_lattices()
    try:
        data = request.get_json()
        target = np.array(data['target'])
        algorithm = data.get('algorithm', 'babai')
        
        nearest = leech_lattice.nearest_neighbor(target, algorithm)
        distance = np.linalg.norm(target - nearest)
        
        return jsonify({
            'nearest_point': nearest.tolist(),
            'distance': float(distance)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@lattice_bp.route('/crypto/encrypt', methods=['POST'])
def crypto_encrypt():
    """Encrypt message using LWE"""
    initialize_lattices()
    try:
        data = request.get_json()
        message = data['message'].encode('utf-8')
        
        ciphertext = lwe_crypto.encrypt(message)
        
        # Convert numpy arrays to lists for JSON serialization
        serialized_ciphertext = []
        for u, v in ciphertext:
            serialized_ciphertext.append({
                'u': u.tolist(),
                'v': int(v)
            })
        
        return jsonify({
            'ciphertext': serialized_ciphertext
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@lattice_bp.route('/crypto/decrypt', methods=['POST'])
def crypto_decrypt():
    """Decrypt ciphertext using LWE"""
    initialize_lattices()
    try:
        data = request.get_json()
        serialized_ciphertext = data['ciphertext']
        
        # Convert back to numpy arrays
        ciphertext = []
        for item in serialized_ciphertext:
            u = np.array(item['u'])
            v = item['v']
            ciphertext.append((u, v))
        
        decrypted = lwe_crypto.decrypt(ciphertext)
        message = decrypted.decode('utf-8', errors='ignore')
        
        return jsonify({
            'message': message
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@lattice_bp.route('/ai/predict', methods=['POST'])
def ai_predict():
    """Make predictions using Equivariant GNN"""
    initialize_lattices()
    try:
        data = request.get_json()
        input_data = np.array(data['input'])
        
        if input_data.ndim == 1:
            input_data = input_data.reshape(1, -1)
        
        predictions = equivariant_gnn.predict(input_data)
        
        return jsonify({
            'predictions': predictions.tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@lattice_bp.route('/ai/train', methods=['POST'])
def ai_train():
    """Train Equivariant GNN"""
    initialize_lattices()
    try:
        data = request.get_json()
        training_data = np.array(data['data'])
        epochs = data.get('epochs', 10)
        learning_rate = data.get('learning_rate', 0.01)
        
        equivariant_gnn.train(training_data, epochs=epochs, learning_rate=learning_rate)
        
        return jsonify({
            'status': 'training_completed',
            'epochs': epochs
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@lattice_bp.route('/theta', methods=['POST'])
def compute_theta():
    """Compute theta function for lattices"""
    initialize_lattices()
    try:
        data = request.get_json()
        lattice_type = data.get('lattice_type', 'e8')
        tau = complex(data.get('tau_real', 1.0), data.get('tau_imag', 1.0))
        max_terms = data.get('max_terms', 100)
        
        if lattice_type.lower() == 'e8':
            theta_value = e8_lattice.compute_theta_function(tau, max_terms)
        elif lattice_type.lower() == 'leech':
            theta_value = leech_lattice.compute_theta_function(tau, max_terms)
        else:
            return jsonify({'error': 'Invalid lattice type'}), 400
        
        return jsonify({
            'theta_value': {
                'real': float(theta_value.real),
                'imag': float(theta_value.imag)
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@lattice_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'E8Leech Framework API',
        'version': '1.0.0'
    })

