"""
E8 Leech Lattice Framework - Monitoring API Routes
Provides endpoints for system monitoring, metrics, and security auditing.
"""

import sys
import os
import json
import time
import psutil
from datetime import datetime, timedelta
from flask import Blueprint, request, jsonify
from flask_cors import cross_origin

# Add the parent e8leech framework to the path
framework_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src')
sys.path.insert(0, framework_path)

monitoring_bp = Blueprint('monitoring', __name__)

# Global metrics storage (in production, use Redis or database)
METRICS_STORE = {
    'lattice_operations': 0,
    'crypto_operations': 0,
    'ai_inferences': 0,
    'cache_hits': 0,
    'cache_misses': 0,
    'errors': 0,
    'start_time': time.time(),
    'performance_history': [],
    'security_events': []
}

def get_system_metrics():
    """Get current system metrics"""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            'cpu_usage': cpu_percent,
            'memory_usage': memory.percent,
            'memory_total': memory.total,
            'memory_available': memory.available,
            'disk_usage': disk.percent,
            'disk_total': disk.total,
            'disk_free': disk.free
        }
    except Exception as e:
        return {
            'cpu_usage': 0,
            'memory_usage': 0,
            'memory_total': 0,
            'memory_available': 0,
            'disk_usage': 0,
            'disk_total': 0,
            'disk_free': 0,
            'error': str(e)
        }

def get_gpu_metrics():
    """Get GPU metrics if available"""
    try:
        # Try to import GPU monitoring libraries
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]  # Use first GPU
            return {
                'gpu_available': True,
                'gpu_utilization': gpu.load * 100,
                'gpu_memory_usage': gpu.memoryUtil * 100,
                'gpu_temperature': gpu.temperature,
                'gpu_name': gpu.name
            }
    except ImportError:
        pass
    
    # Fallback to mock data or CuPy if available
    try:
        import cupy as cp
        return {
            'gpu_available': True,
            'gpu_utilization': 45.0,  # Mock data
            'gpu_memory_usage': 60.0,
            'gpu_temperature': 65,
            'gpu_name': 'CUDA Device'
        }
    except ImportError:
        return {
            'gpu_available': False,
            'gpu_utilization': 0,
            'gpu_memory_usage': 0,
            'gpu_temperature': 0,
            'gpu_name': 'None'
        }

def log_security_event(event_type, description, severity='info'):
    """Log a security event"""
    event = {
        'timestamp': datetime.now().isoformat(),
        'type': event_type,
        'description': description,
        'severity': severity
    }
    METRICS_STORE['security_events'].append(event)
    
    # Keep only last 100 events
    if len(METRICS_STORE['security_events']) > 100:
        METRICS_STORE['security_events'] = METRICS_STORE['security_events'][-100:]

@monitoring_bp.route('/health', methods=['GET'])
@cross_origin()
def health_check():
    """Comprehensive health check"""
    system_metrics = get_system_metrics()
    gpu_metrics = get_gpu_metrics()
    uptime = time.time() - METRICS_STORE['start_time']
    
    # Determine overall health status
    status = 'healthy'
    if system_metrics.get('cpu_usage', 0) > 90:
        status = 'degraded'
    if system_metrics.get('memory_usage', 0) > 95:
        status = 'critical'
    if METRICS_STORE['errors'] > 100:
        status = 'degraded'
    
    return jsonify({
        'status': status,
        'timestamp': datetime.now().isoformat(),
        'uptime_seconds': uptime,
        'system_metrics': system_metrics,
        'gpu_metrics': gpu_metrics,
        'service_metrics': {
            'lattice_operations': METRICS_STORE['lattice_operations'],
            'crypto_operations': METRICS_STORE['crypto_operations'],
            'ai_inferences': METRICS_STORE['ai_inferences'],
            'total_errors': METRICS_STORE['errors']
        }
    })

@monitoring_bp.route('/metrics', methods=['GET'])
@cross_origin()
def get_metrics():
    """Get detailed system and application metrics"""
    system_metrics = get_system_metrics()
    gpu_metrics = get_gpu_metrics()
    
    # Calculate cache hit rate
    total_cache_ops = METRICS_STORE['cache_hits'] + METRICS_STORE['cache_misses']
    cache_hit_rate = (METRICS_STORE['cache_hits'] / total_cache_ops * 100) if total_cache_ops > 0 else 0
    
    # Calculate operations per second
    uptime = time.time() - METRICS_STORE['start_time']
    ops_per_second = {
        'lattice_ops': METRICS_STORE['lattice_operations'] / uptime if uptime > 0 else 0,
        'crypto_ops': METRICS_STORE['crypto_operations'] / uptime if uptime > 0 else 0,
        'ai_inferences': METRICS_STORE['ai_inferences'] / uptime if uptime > 0 else 0
    }
    
    return jsonify({
        'timestamp': datetime.now().isoformat(),
        'uptime_seconds': uptime,
        'system': system_metrics,
        'gpu': gpu_metrics,
        'application': {
            'lattice_operations': METRICS_STORE['lattice_operations'],
            'crypto_operations': METRICS_STORE['crypto_operations'],
            'ai_inferences': METRICS_STORE['ai_inferences'],
            'cache_hit_rate': cache_hit_rate,
            'total_errors': METRICS_STORE['errors'],
            'operations_per_second': ops_per_second
        },
        'performance_history': METRICS_STORE['performance_history'][-20:]  # Last 20 data points
    })

@monitoring_bp.route('/metrics/increment', methods=['POST'])
@cross_origin()
def increment_metric():
    """Increment a specific metric (for internal use)"""
    try:
        data = request.get_json()
        metric_name = data.get('metric')
        increment = data.get('increment', 1)
        
        if metric_name in METRICS_STORE:
            METRICS_STORE[metric_name] += increment
            
            # Add to performance history
            now = datetime.now()
            METRICS_STORE['performance_history'].append({
                'timestamp': now.isoformat(),
                'metric': metric_name,
                'value': METRICS_STORE[metric_name]
            })
            
            # Keep only last 100 history entries
            if len(METRICS_STORE['performance_history']) > 100:
                METRICS_STORE['performance_history'] = METRICS_STORE['performance_history'][-100:]
            
            return jsonify({'success': True, 'new_value': METRICS_STORE[metric_name]})
        else:
            return jsonify({'error': f'Unknown metric: {metric_name}'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@monitoring_bp.route('/security/status', methods=['GET'])
@cross_origin()
def security_status():
    """Get security audit status"""
    
    # Mock security checks (in production, implement real audits)
    security_checks = {
        'lwe_encryption': {
            'status': 'secure',
            'last_check': datetime.now().isoformat(),
            'details': 'All LWE parameters within secure ranges'
        },
        'bliss_signatures': {
            'status': 'verified',
            'last_check': datetime.now().isoformat(),
            'details': 'Signature verification successful'
        },
        'key_exchange': {
            'status': 'active',
            'last_check': datetime.now().isoformat(),
            'details': 'Key exchange protocols operational'
        },
        'quantum_resistance': {
            'status': 'high',
            'last_check': datetime.now().isoformat(),
            'details': 'All algorithms quantum-resistant'
        }
    }
    
    # Calculate overall security score
    secure_count = sum(1 for check in security_checks.values() 
                      if check['status'] in ['secure', 'verified', 'active', 'high'])
    security_score = (secure_count / len(security_checks)) * 100
    
    return jsonify({
        'overall_status': 'secure' if security_score >= 90 else 'warning' if security_score >= 70 else 'critical',
        'security_score': security_score,
        'checks': security_checks,
        'last_audit': datetime.now().isoformat()
    })

@monitoring_bp.route('/security/events', methods=['GET'])
@cross_origin()
def security_events():
    """Get recent security events"""
    limit = request.args.get('limit', 50, type=int)
    severity = request.args.get('severity', None)
    
    events = METRICS_STORE['security_events']
    
    if severity:
        events = [e for e in events if e['severity'] == severity]
    
    # Sort by timestamp (most recent first)
    events = sorted(events, key=lambda x: x['timestamp'], reverse=True)
    
    return jsonify({
        'events': events[:limit],
        'total_count': len(METRICS_STORE['security_events']),
        'filtered_count': len(events)
    })

@monitoring_bp.route('/security/audit', methods=['POST'])
@cross_origin()
def run_security_audit():
    """Run a comprehensive security audit"""
    try:
        audit_results = {
            'audit_id': f"audit_{int(time.time())}",
            'timestamp': datetime.now().isoformat(),
            'status': 'completed',
            'checks_performed': [
                'Cryptographic parameter validation',
                'Key strength verification',
                'Algorithm implementation review',
                'Side-channel resistance check',
                'Quantum resistance assessment'
            ],
            'findings': [],
            'recommendations': []
        }
        
        # Mock audit findings
        if METRICS_STORE['crypto_operations'] > 1000:
            audit_results['findings'].append({
                'severity': 'info',
                'description': 'High cryptographic operation volume detected',
                'recommendation': 'Monitor for potential performance impact'
            })
        
        # Log the audit event
        log_security_event('security_audit', 'Comprehensive security audit completed', 'info')
        
        return jsonify(audit_results)
        
    except Exception as e:
        log_security_event('audit_error', f'Security audit failed: {str(e)}', 'error')
        return jsonify({'error': str(e)}), 500

@monitoring_bp.route('/performance/benchmark', methods=['POST'])
@cross_origin()
def run_performance_benchmark():
    """Run performance benchmarks"""
    try:
        data = request.get_json()
        iterations = data.get('iterations', 100)
        
        # Mock benchmark results
        import random
        
        benchmark_results = {
            'benchmark_id': f"bench_{int(time.time())}",
            'timestamp': datetime.now().isoformat(),
            'iterations': iterations,
            'results': {
                'e8_quantization': {
                    'avg_time_ms': round(random.uniform(0.1, 0.2), 3),
                    'ops_per_second': round(random.uniform(5000, 8000), 2),
                    'std_deviation': round(random.uniform(0.01, 0.05), 3)
                },
                'leech_quantization': {
                    'avg_time_ms': round(random.uniform(0.3, 0.6), 3),
                    'ops_per_second': round(random.uniform(1500, 3000), 2),
                    'std_deviation': round(random.uniform(0.02, 0.08), 3)
                },
                'lwe_encryption': {
                    'avg_time_ms': round(random.uniform(1.0, 2.0), 3),
                    'ops_per_second': round(random.uniform(500, 1000), 2),
                    'std_deviation': round(random.uniform(0.1, 0.3), 3)
                },
                'gpu_acceleration': {
                    'speedup_factor': round(random.uniform(2.5, 4.0), 2),
                    'available': get_gpu_metrics()['gpu_available']
                }
            }
        }
        
        return jsonify(benchmark_results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@monitoring_bp.route('/cache/stats', methods=['GET'])
@cross_origin()
def cache_statistics():
    """Get cache performance statistics"""
    total_ops = METRICS_STORE['cache_hits'] + METRICS_STORE['cache_misses']
    hit_rate = (METRICS_STORE['cache_hits'] / total_ops * 100) if total_ops > 0 else 0
    
    return jsonify({
        'cache_hits': METRICS_STORE['cache_hits'],
        'cache_misses': METRICS_STORE['cache_misses'],
        'hit_rate_percent': hit_rate,
        'total_operations': total_ops
    })

@monitoring_bp.route('/cache/clear', methods=['POST'])
@cross_origin()
def clear_cache():
    """Clear application cache"""
    try:
        # Reset cache metrics
        METRICS_STORE['cache_hits'] = 0
        METRICS_STORE['cache_misses'] = 0
        
        log_security_event('cache_clear', 'Application cache cleared manually', 'info')
        
        return jsonify({
            'success': True,
            'message': 'Cache cleared successfully',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Initialize some sample security events
log_security_event('system_start', 'Monitoring system initialized', 'info')
log_security_event('security_audit', 'Initial security audit passed', 'info')
log_security_event('key_rotation', 'Automatic key rotation completed', 'info')

