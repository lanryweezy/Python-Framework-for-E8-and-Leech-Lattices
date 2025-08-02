"""
Security Monitor for E8 Leech Lattice Framework
Provides real-time security monitoring and alerting.
"""

import time
import threading
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

class SecurityMonitor:
    """Real-time security monitoring system"""
    
    def __init__(self, check_interval: float = 60.0):
        self.check_interval = check_interval
        self.monitoring = False
        self.monitor_thread = None
        self.alert_callbacks = []
        self.metrics = {
            'failed_operations': 0,
            'security_violations': 0,
            'anomalies_detected': 0,
            'last_audit_time': None,
            'system_health': 'unknown'
        }
        self.thresholds = {
            'max_failed_operations_per_hour': 100,
            'max_security_violations_per_hour': 10,
            'max_anomalies_per_hour': 50,
            'min_audit_frequency_hours': 24
        }
        self.event_history = []
        self.lock = threading.Lock()
    
    def start_monitoring(self):
        """Start the security monitoring system"""
        if self.monitoring:
            logger.warning("Security monitoring is already running")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Security monitoring started")
    
    def stop_monitoring(self):
        """Stop the security monitoring system"""
        if not self.monitoring:
            logger.warning("Security monitoring is not running")
            return
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("Security monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                self._perform_security_checks()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in security monitoring loop: {e}")
                time.sleep(self.check_interval)
    
    def _perform_security_checks(self):
        """Perform periodic security checks"""
        current_time = datetime.now()
        
        # Check system health
        health_status = self._check_system_health()
        
        # Check for threshold violations
        violations = self._check_thresholds()
        
        # Check audit compliance
        audit_status = self._check_audit_compliance()
        
        # Check for anomalies
        anomalies = self._detect_anomalies()
        
        # Update metrics
        with self.lock:
            self.metrics['system_health'] = health_status
            if violations:
                self.metrics['security_violations'] += len(violations)
            if anomalies:
                self.metrics['anomalies_detected'] += len(anomalies)
        
        # Generate alerts if necessary
        if violations or anomalies or health_status == 'critical':
            self._generate_alerts(violations, anomalies, health_status)
        
        # Log monitoring event
        self._log_monitoring_event(current_time, health_status, violations, anomalies)
    
    def _check_system_health(self) -> str:
        """Check overall system health"""
        try:
            # Check CPU and memory usage
            import psutil
            cpu_usage = psutil.cpu_percent(interval=1)
            memory_usage = psutil.virtual_memory().percent
            
            # Determine health status
            if cpu_usage > 90 or memory_usage > 95:
                return 'critical'
            elif cpu_usage > 80 or memory_usage > 85:
                return 'degraded'
            else:
                return 'healthy'
                
        except ImportError:
            # Fallback if psutil not available
            return 'unknown'
        except Exception as e:
            logger.error(f"Error checking system health: {e}")
            return 'error'
    
    def _check_thresholds(self) -> List[Dict[str, Any]]:
        """Check for threshold violations"""
        violations = []
        current_time = datetime.now()
        one_hour_ago = current_time - timedelta(hours=1)
        
        # Count recent events
        recent_events = [
            event for event in self.event_history
            if datetime.fromisoformat(event['timestamp']) > one_hour_ago
        ]
        
        failed_ops = len([e for e in recent_events if e.get('type') == 'failed_operation'])
        security_violations = len([e for e in recent_events if e.get('type') == 'security_violation'])
        anomalies = len([e for e in recent_events if e.get('type') == 'anomaly'])
        
        # Check thresholds
        if failed_ops > self.thresholds['max_failed_operations_per_hour']:
            violations.append({
                'type': 'failed_operations_threshold',
                'current': failed_ops,
                'threshold': self.thresholds['max_failed_operations_per_hour'],
                'severity': 'high'
            })
        
        if security_violations > self.thresholds['max_security_violations_per_hour']:
            violations.append({
                'type': 'security_violations_threshold',
                'current': security_violations,
                'threshold': self.thresholds['max_security_violations_per_hour'],
                'severity': 'critical'
            })
        
        if anomalies > self.thresholds['max_anomalies_per_hour']:
            violations.append({
                'type': 'anomalies_threshold',
                'current': anomalies,
                'threshold': self.thresholds['max_anomalies_per_hour'],
                'severity': 'medium'
            })
        
        return violations
    
    def _check_audit_compliance(self) -> Dict[str, Any]:
        """Check audit compliance"""
        if not self.metrics['last_audit_time']:
            return {
                'compliant': False,
                'reason': 'No audit has been performed',
                'severity': 'high'
            }
        
        last_audit = datetime.fromisoformat(self.metrics['last_audit_time'])
        hours_since_audit = (datetime.now() - last_audit).total_seconds() / 3600
        
        if hours_since_audit > self.thresholds['min_audit_frequency_hours']:
            return {
                'compliant': False,
                'reason': f'Last audit was {hours_since_audit:.1f} hours ago',
                'severity': 'medium'
            }
        
        return {'compliant': True}
    
    def _detect_anomalies(self) -> List[Dict[str, Any]]:
        """Detect security anomalies"""
        anomalies = []
        
        # Check for unusual patterns in recent events
        current_time = datetime.now()
        recent_events = [
            event for event in self.event_history[-100:]  # Last 100 events
            if (current_time - datetime.fromisoformat(event['timestamp'])).total_seconds() < 3600
        ]
        
        if not recent_events:
            return anomalies
        
        # Check for frequency anomalies
        event_types = {}
        for event in recent_events:
            event_type = event.get('operation', 'unknown')
            event_types[event_type] = event_types.get(event_type, 0) + 1
        
        # Detect spikes in specific operations
        for event_type, count in event_types.items():
            if count > 50:  # More than 50 operations of same type in an hour
                anomalies.append({
                    'type': 'frequency_anomaly',
                    'operation': event_type,
                    'count': count,
                    'severity': 'medium',
                    'description': f'High frequency of {event_type} operations: {count} in last hour'
                })
        
        # Check for error rate anomalies
        error_events = [e for e in recent_events if not e.get('success', True)]
        if len(recent_events) > 0:
            error_rate = len(error_events) / len(recent_events)
            if error_rate > 0.1:  # More than 10% error rate
                anomalies.append({
                    'type': 'error_rate_anomaly',
                    'error_rate': error_rate,
                    'severity': 'high',
                    'description': f'High error rate: {error_rate:.1%}'
                })
        
        return anomalies
    
    def _generate_alerts(self, violations: List[Dict[str, Any]], 
                        anomalies: List[Dict[str, Any]], 
                        health_status: str):
        """Generate security alerts"""
        alerts = []
        
        # Create alerts for violations
        for violation in violations:
            alerts.append({
                'timestamp': datetime.now().isoformat(),
                'type': 'threshold_violation',
                'severity': violation['severity'],
                'message': f"Threshold violation: {violation['type']}",
                'details': violation
            })
        
        # Create alerts for anomalies
        for anomaly in anomalies:
            alerts.append({
                'timestamp': datetime.now().isoformat(),
                'type': 'anomaly_detected',
                'severity': anomaly['severity'],
                'message': f"Anomaly detected: {anomaly['type']}",
                'details': anomaly
            })
        
        # Create alert for critical health status
        if health_status == 'critical':
            alerts.append({
                'timestamp': datetime.now().isoformat(),
                'type': 'system_health',
                'severity': 'critical',
                'message': 'System health is critical',
                'details': {'health_status': health_status}
            })
        
        # Send alerts to registered callbacks
        for alert in alerts:
            self._send_alert(alert)
    
    def _send_alert(self, alert: Dict[str, Any]):
        """Send alert to registered callbacks"""
        logger.warning(f"Security alert: {alert['message']}")
        
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    def _log_monitoring_event(self, timestamp: datetime, health_status: str,
                             violations: List[Dict[str, Any]], 
                             anomalies: List[Dict[str, Any]]):
        """Log monitoring event"""
        event = {
            'timestamp': timestamp.isoformat(),
            'type': 'monitoring_check',
            'health_status': health_status,
            'violations_count': len(violations),
            'anomalies_count': len(anomalies),
            'violations': violations,
            'anomalies': anomalies
        }
        
        with self.lock:
            self.event_history.append(event)
            
            # Keep only last 1000 events
            if len(self.event_history) > 1000:
                self.event_history = self.event_history[-1000:]
    
    def register_alert_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Register an alert callback function"""
        self.alert_callbacks.append(callback)
        logger.info("Registered new alert callback")
    
    def log_security_event(self, event_type: str, operation: str, 
                          success: bool = True, details: Optional[Dict[str, Any]] = None):
        """Log a security event"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'type': event_type,
            'operation': operation,
            'success': success,
            'details': details or {}
        }
        
        with self.lock:
            self.event_history.append(event)
            
            # Update metrics
            if not success:
                self.metrics['failed_operations'] += 1
            
            if event_type == 'security_violation':
                self.metrics['security_violations'] += 1
    
    def update_audit_time(self, audit_time: Optional[str] = None):
        """Update the last audit time"""
        with self.lock:
            self.metrics['last_audit_time'] = audit_time or datetime.now().isoformat()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current security metrics"""
        with self.lock:
            return self.metrics.copy()
    
    def get_recent_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent security events"""
        with self.lock:
            return self.event_history[-limit:]
    
    def update_threshold(self, threshold_name: str, value: Any):
        """Update a monitoring threshold"""
        if threshold_name in self.thresholds:
            old_value = self.thresholds[threshold_name]
            self.thresholds[threshold_name] = value
            logger.info(f"Updated threshold {threshold_name}: {old_value} -> {value}")
        else:
            logger.warning(f"Unknown threshold: {threshold_name}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get monitoring system status"""
        with self.lock:
            return {
                'monitoring_active': self.monitoring,
                'check_interval': self.check_interval,
                'metrics': self.metrics.copy(),
                'thresholds': self.thresholds.copy(),
                'alert_callbacks_count': len(self.alert_callbacks),
                'event_history_size': len(self.event_history)
            }

