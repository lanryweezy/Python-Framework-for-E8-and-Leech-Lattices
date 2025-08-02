"""
Security Hooks for E8 Leech Lattice Framework
Provides security monitoring hooks for cryptographic operations.
"""

import time
import functools
from typing import Callable, Any, Dict, List, Optional
from datetime import datetime
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

class SecurityHooks:
    """Security monitoring hooks for cryptographic operations"""
    
    def __init__(self):
        self.hooks = {}
        self.event_log = []
        self.security_policies = {
            'max_operation_time': 10.0,  # seconds
            'max_key_reuse': 1000,       # operations
            'require_audit_trail': True,
            'alert_on_anomalies': True
        }
        self.operation_counts = {}
        self.anomaly_thresholds = {
            'timing_deviation': 3.0,     # standard deviations
            'frequency_spike': 5.0,      # times normal rate
            'error_rate': 0.05           # 5% error rate threshold
        }
    
    def register_hook(self, operation: str, hook_func: Callable):
        """Register a security hook for an operation"""
        if operation not in self.hooks:
            self.hooks[operation] = []
        self.hooks[operation].append(hook_func)
        logger.info(f"Registered security hook for operation: {operation}")
    
    def security_monitor(self, operation: str, log_params: bool = True):
        """Decorator for monitoring cryptographic operations"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                operation_id = f"{operation}_{int(start_time * 1000000)}"
                
                # Pre-operation security checks
                self._pre_operation_checks(operation, operation_id, args, kwargs)
                
                try:
                    # Execute the operation
                    result = func(*args, **kwargs)
                    
                    # Post-operation security checks
                    end_time = time.time()
                    execution_time = end_time - start_time
                    
                    self._post_operation_checks(
                        operation, operation_id, execution_time, 
                        result, args, kwargs, success=True
                    )
                    
                    return result
                    
                except Exception as e:
                    # Handle operation failure
                    end_time = time.time()
                    execution_time = end_time - start_time
                    
                    self._post_operation_checks(
                        operation, operation_id, execution_time,
                        None, args, kwargs, success=False, error=str(e)
                    )
                    
                    raise
            
            return wrapper
        return decorator
    
    def _pre_operation_checks(self, operation: str, operation_id: str, args: tuple, kwargs: dict):
        """Perform pre-operation security checks"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'operation_id': operation_id,
            'operation': operation,
            'phase': 'pre_operation',
            'checks': []
        }
        
        # Check operation frequency
        current_time = time.time()
        if operation not in self.operation_counts:
            self.operation_counts[operation] = []
        
        # Clean old entries (last hour)
        self.operation_counts[operation] = [
            t for t in self.operation_counts[operation] 
            if current_time - t < 3600
        ]
        
        # Check for frequency anomalies
        recent_ops = len([
            t for t in self.operation_counts[operation] 
            if current_time - t < 60  # last minute
        ])
        
        if recent_ops > 100:  # More than 100 ops per minute
            event['checks'].append({
                'check': 'frequency_anomaly',
                'status': 'warning',
                'details': f"High operation frequency: {recent_ops} ops/min"
            })
            self._trigger_security_alert('frequency_anomaly', operation, recent_ops)
        
        # Check for parameter anomalies
        param_check = self._check_parameters(operation, args, kwargs)
        if param_check:
            event['checks'].append(param_check)
        
        # Execute registered hooks
        if operation in self.hooks:
            for hook in self.hooks[operation]:
                try:
                    hook_result = hook('pre', operation_id, args, kwargs)
                    if hook_result:
                        event['checks'].append(hook_result)
                except Exception as e:
                    logger.error(f"Security hook failed for {operation}: {e}")
        
        self.event_log.append(event)
        self.operation_counts[operation].append(current_time)
    
    def _post_operation_checks(self, operation: str, operation_id: str, 
                              execution_time: float, result: Any, 
                              args: tuple, kwargs: dict, success: bool, 
                              error: Optional[str] = None):
        """Perform post-operation security checks"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'operation_id': operation_id,
            'operation': operation,
            'phase': 'post_operation',
            'execution_time': execution_time,
            'success': success,
            'checks': []
        }
        
        if error:
            event['error'] = error
        
        # Check execution time
        if execution_time > self.security_policies['max_operation_time']:
            event['checks'].append({
                'check': 'timing_anomaly',
                'status': 'warning',
                'details': f"Operation took {execution_time:.3f}s (max: {self.security_policies['max_operation_time']}s)"
            })
            self._trigger_security_alert('timing_anomaly', operation, execution_time)
        
        # Check for timing attacks
        timing_check = self._check_timing_patterns(operation, execution_time)
        if timing_check:
            event['checks'].append(timing_check)
        
        # Check result integrity
        if success and result is not None:
            integrity_check = self._check_result_integrity(operation, result)
            if integrity_check:
                event['checks'].append(integrity_check)
        
        # Execute registered hooks
        if operation in self.hooks:
            for hook in self.hooks[operation]:
                try:
                    hook_result = hook('post', operation_id, result, execution_time, success)
                    if hook_result:
                        event['checks'].append(hook_result)
                except Exception as e:
                    logger.error(f"Security hook failed for {operation}: {e}")
        
        self.event_log.append(event)
        
        # Cleanup old events (keep last 1000)
        if len(self.event_log) > 1000:
            self.event_log = self.event_log[-1000:]
    
    def _check_parameters(self, operation: str, args: tuple, kwargs: dict) -> Optional[Dict[str, Any]]:
        """Check operation parameters for security issues"""
        checks = []
        
        # Check for common parameter issues
        if operation in ['lwe_encrypt', 'lwe_decrypt']:
            # Check LWE parameters
            if 'n' in kwargs and kwargs['n'] < 256:
                checks.append("LWE dimension too small for security")
            if 'q' in kwargs and kwargs['q'] < 2**15:
                checks.append("LWE modulus too small")
        
        elif operation in ['bliss_sign', 'bliss_verify']:
            # Check BLISS parameters
            if 'n' in kwargs and kwargs['n'] < 512:
                checks.append("BLISS dimension too small")
        
        # Check for null or empty parameters
        for i, arg in enumerate(args):
            if arg is None:
                checks.append(f"Null argument at position {i}")
        
        for key, value in kwargs.items():
            if value is None:
                checks.append(f"Null parameter: {key}")
        
        if checks:
            return {
                'check': 'parameter_validation',
                'status': 'warning',
                'details': '; '.join(checks)
            }
        
        return None
    
    def _check_timing_patterns(self, operation: str, execution_time: float) -> Optional[Dict[str, Any]]:
        """Check for timing attack patterns"""
        # Get recent timing data for this operation
        recent_events = [
            event for event in self.event_log[-100:]  # Last 100 events
            if event.get('operation') == operation and 
               event.get('phase') == 'post_operation' and
               'execution_time' in event
        ]
        
        if len(recent_events) < 10:
            return None  # Not enough data
        
        # Calculate timing statistics
        times = [event['execution_time'] for event in recent_events]
        mean_time = sum(times) / len(times)
        variance = sum((t - mean_time) ** 2 for t in times) / len(times)
        std_dev = variance ** 0.5
        
        # Check for timing anomalies
        if std_dev > 0 and abs(execution_time - mean_time) > self.anomaly_thresholds['timing_deviation'] * std_dev:
            return {
                'check': 'timing_pattern',
                'status': 'warning',
                'details': f"Timing deviation: {abs(execution_time - mean_time):.3f}s from mean {mean_time:.3f}s"
            }
        
        return None
    
    def _check_result_integrity(self, operation: str, result: Any) -> Optional[Dict[str, Any]]:
        """Check result integrity"""
        checks = []
        
        # Basic result validation
        if result is None:
            checks.append("Operation returned null result")
        
        # Operation-specific checks
        if operation in ['lwe_encrypt', 'bliss_sign']:
            # Check if result looks like valid ciphertext/signature
            if hasattr(result, '__len__') and len(result) == 0:
                checks.append("Empty result from cryptographic operation")
        
        elif operation in ['quantize', 'nearest_neighbor']:
            # Check lattice operation results
            if hasattr(result, 'shape') and len(result.shape) == 0:
                checks.append("Scalar result from vector operation")
        
        if checks:
            return {
                'check': 'result_integrity',
                'status': 'warning',
                'details': '; '.join(checks)
            }
        
        return None
    
    def _trigger_security_alert(self, alert_type: str, operation: str, details: Any):
        """Trigger a security alert"""
        if not self.security_policies['alert_on_anomalies']:
            return
        
        alert = {
            'timestamp': datetime.now().isoformat(),
            'type': alert_type,
            'operation': operation,
            'details': details,
            'severity': self._determine_alert_severity(alert_type)
        }
        
        logger.warning(f"Security alert: {alert_type} in {operation} - {details}")
        
        # In a production system, this would send alerts to monitoring systems
        # For now, we just log it
    
    def _determine_alert_severity(self, alert_type: str) -> str:
        """Determine alert severity"""
        severity_map = {
            'frequency_anomaly': 'medium',
            'timing_anomaly': 'low',
            'parameter_anomaly': 'high',
            'result_anomaly': 'high',
            'authentication_failure': 'critical',
            'unauthorized_access': 'critical'
        }
        return severity_map.get(alert_type, 'medium')
    
    def get_security_events(self, operation: Optional[str] = None, 
                           limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get security events"""
        events = self.event_log
        
        if operation:
            events = [e for e in events if e.get('operation') == operation]
        
        # Sort by timestamp (most recent first)
        events = sorted(events, key=lambda x: x['timestamp'], reverse=True)
        
        return events[:limit] if limit else events
    
    def get_operation_statistics(self) -> Dict[str, Any]:
        """Get operation statistics"""
        stats = {}
        
        for operation, timestamps in self.operation_counts.items():
            current_time = time.time()
            recent_ops = [t for t in timestamps if current_time - t < 3600]  # Last hour
            
            stats[operation] = {
                'total_operations': len(timestamps),
                'operations_last_hour': len(recent_ops),
                'average_frequency': len(recent_ops) / 60 if recent_ops else 0,  # ops per minute
                'last_operation': max(timestamps) if timestamps else None
            }
        
        return stats
    
    def clear_event_log(self):
        """Clear the security event log"""
        self.event_log.clear()
        logger.info("Security event log cleared")
    
    def update_security_policy(self, policy: str, value: Any):
        """Update a security policy"""
        if policy in self.security_policies:
            old_value = self.security_policies[policy]
            self.security_policies[policy] = value
            logger.info(f"Updated security policy {policy}: {old_value} -> {value}")
        else:
            logger.warning(f"Unknown security policy: {policy}")
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security monitoring summary"""
        current_time = time.time()
        recent_events = [
            e for e in self.event_log 
            if current_time - time.mktime(time.strptime(e['timestamp'][:19], '%Y-%m-%dT%H:%M:%S')) < 3600
        ]
        
        alerts = [
            e for e in recent_events 
            if any(check.get('status') in ['warning', 'error'] for check in e.get('checks', []))
        ]
        
        return {
            'total_events': len(self.event_log),
            'events_last_hour': len(recent_events),
            'alerts_last_hour': len(alerts),
            'monitored_operations': list(self.operation_counts.keys()),
            'active_hooks': {op: len(hooks) for op, hooks in self.hooks.items()},
            'security_policies': self.security_policies
        }

