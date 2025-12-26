#!/usr/bin/env python3

"""
Pipeline Monitor Node for Vision-Language-Action (VLA) Module
Monitors the complete VLA pipeline and provides performance metrics
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
import time
from typing import Dict, List, Any, Optional
import json
from collections import defaultdict, deque
import threading

# Import custom message types
from vla_integration.msg import ActionPlan, Task, ExecutionStatus
from vla_integration.srv import ExecutePlan


class PipelineMonitorNode(Node):
    """
    ROS 2 Node for monitoring the complete VLA pipeline and providing performance metrics
    """

    def __init__(self):
        super().__init__('pipeline_monitor')

        # Declare parameters
        self.declare_parameter('monitoring_rate', 1.0)  # Hz
        self.declare_parameter('metrics_history_size', 100)
        self.declare_parameter('performance_thresholds', {
            'max_execution_time': 300.0,  # seconds
            'min_success_rate': 0.8,      # 80%
            'max_error_rate': 0.2,        # 20%
        })
        self.declare_parameter('logging_enabled', True)

        # Get parameters
        self.monitoring_rate = self.get_parameter('monitoring_rate').value
        self.metrics_history_size = self.get_parameter('metrics_history_size').value
        self.performance_thresholds = self.get_parameter('performance_thresholds').value
        self.logging_enabled = self.get_parameter('logging_enabled').value

        # Create subscribers for pipeline status
        self.pipeline_status_sub = self.create_subscription(
            ExecutionStatus,
            'vla/orchestration_status',
            self.pipeline_status_callback,
            10
        )

        self.execution_status_sub = self.create_subscription(
            ExecutionStatus,
            'vla/execution_status',
            self.execution_status_callback,
            10
        )

        self.voice_status_sub = self.create_subscription(
            ExecutionStatus,
            'vla/voice_to_action_status',
            self.voice_status_callback,
            10
        )

        # Create publisher for monitoring metrics
        self.metrics_pub = self.create_publisher(ExecutionStatus, 'vla/pipeline_metrics', 10)

        # Initialize monitoring state
        self.execution_metrics: Dict[str, Dict[str, Any]] = {}
        self.pipeline_history: List[Dict[str, Any]] = []
        self.metrics_history: Dict[str, deque] = {
            'execution_time': deque(maxlen=self.metrics_history_size),
            'success_rate': deque(maxlen=self.metrics_history_size),
            'error_rate': deque(maxlen=self.metrics_history_size),
            'throughput': deque(maxlen=self.metrics_history_size)
        }
        self.component_status: Dict[str, Dict[str, Any]] = {}
        self.monitoring_lock = threading.Lock()

        # Create timer for monitoring
        self.monitor_timer = self.create_timer(1.0/self.monitoring_rate, self.monitoring_callback)

        self.get_logger().info('Pipeline Monitor Node initialized')

    def pipeline_status_callback(self, msg: ExecutionStatus):
        """
        Callback for pipeline status messages
        """
        try:
            with self.monitoring_lock:
                self.update_execution_metrics(msg)
                self.log_pipeline_event('pipeline_status', msg)

                if self.logging_enabled:
                    self.get_logger().debug(f'Pipeline status: {msg.execution_id}, status: {msg.overall_status}, progress: {msg.progress}')

        except Exception as e:
            self.get_logger().error(f'Error in pipeline status callback: {str(e)}')

    def execution_status_callback(self, msg: ExecutionStatus):
        """
        Callback for execution status messages
        """
        try:
            with self.monitoring_lock:
                self.update_execution_metrics(msg)
                self.log_pipeline_event('execution_status', msg)

                if self.logging_enabled:
                    self.get_logger().debug(f'Execution status: {msg.execution_id}, status: {msg.overall_status}, progress: {msg.progress}')

        except Exception as e:
            self.get_logger().error(f'Error in execution status callback: {str(e)}')

    def voice_status_callback(self, msg: ExecutionStatus):
        """
        Callback for voice status messages
        """
        try:
            with self.monitoring_lock:
                self.update_execution_metrics(msg)
                self.log_pipeline_event('voice_status', msg)

                if self.logging_enabled:
                    self.get_logger().debug(f'Voice status: {msg.execution_id}, status: {msg.overall_status}, progress: {msg.progress}')

        except Exception as e:
            self.get_logger().error(f'Error in voice status callback: {str(e)}')

    def update_execution_metrics(self, status_msg: ExecutionStatus):
        """
        Update execution metrics based on status message
        """
        try:
            execution_id = status_msg.execution_id

            if execution_id not in self.execution_metrics:
                self.execution_metrics[execution_id] = {
                    'start_time': time.time(),
                    'start_timestamp': status_msg.header.stamp.sec + status_msg.header.stamp.nanosec / 1e9,
                    'status_history': [],
                    'current_status': status_msg.overall_status,
                    'progress_history': [],
                    'last_update': time.time(),
                    'completed_tasks': status_msg.completed_tasks,
                    'total_tasks': status_msg.total_tasks
                }

            # Update metrics
            metrics = self.execution_metrics[execution_id]
            metrics['current_status'] = status_msg.overall_status
            metrics['last_update'] = time.time()
            metrics['completed_tasks'] = status_msg.completed_tasks
            metrics['total_tasks'] = status_msg.total_tasks

            # Add to status history
            metrics['status_history'].append({
                'status': status_msg.overall_status,
                'progress': status_msg.progress,
                'timestamp': time.time()
            })

            # Add to progress history
            metrics['progress_history'].append({
                'progress': status_msg.progress,
                'timestamp': time.time()
            })

            # If execution is complete, calculate final metrics
            if status_msg.overall_status in ['completed', 'failed', 'error']:
                self.calculate_final_metrics(execution_id)

        except Exception as e:
            self.get_logger().error(f'Error updating execution metrics: {str(e)}')

    def calculate_final_metrics(self, execution_id: str):
        """
        Calculate final metrics for a completed execution
        """
        try:
            if execution_id not in self.execution_metrics:
                return

            metrics = self.execution_metrics[execution_id]
            start_time = metrics['start_time']
            end_time = time.time()
            execution_time = end_time - start_time

            # Calculate success based on final status
            success = metrics['current_status'] == 'completed'
            error = metrics['current_status'] in ['failed', 'error']

            # Add to history metrics
            self.metrics_history['execution_time'].append(execution_time)

            # Calculate success and error rates based on recent history
            recent_executions = list(self.execution_metrics.values())[-10:]  # Last 10 executions
            if recent_executions:
                recent_successes = sum(1 for exec_metrics in recent_executions
                                     if exec_metrics['current_status'] == 'completed')
                recent_errors = sum(1 for exec_metrics in recent_executions
                                  if exec_metrics['current_status'] in ['failed', 'error'])

                success_rate = recent_successes / len(recent_executions)
                error_rate = recent_errors / len(recent_executions)

                self.metrics_history['success_rate'].append(success_rate)
                self.metrics_history['error_rate'].append(error_rate)

            # Calculate throughput (executions per minute)
            if len(self.pipeline_history) > 0:
                time_span = end_time - self.pipeline_history[0].get('start_time', end_time)
                if time_span > 0:
                    throughput = len(self.pipeline_history) / (time_span / 60)  # per minute
                    self.metrics_history['throughput'].append(throughput)

            # Add to pipeline history
            pipeline_entry = {
                'execution_id': execution_id,
                'start_time': start_time,
                'end_time': end_time,
                'execution_time': execution_time,
                'success': success,
                'error': error,
                'final_status': metrics['current_status'],
                'completed_tasks': metrics['completed_tasks'],
                'total_tasks': metrics['total_tasks']
            }

            self.pipeline_history.append(pipeline_entry)

            # Keep pipeline history size manageable
            if len(self.pipeline_history) > self.metrics_history_size:
                self.pipeline_history = self.pipeline_history[-self.metrics_history_size:]

            self.get_logger().info(f'Execution {execution_id} completed in {execution_time:.2f}s, success: {success}')

        except Exception as e:
            self.get_logger().error(f'Error calculating final metrics: {str(e)}')

    def monitoring_callback(self):
        """
        Main monitoring callback that calculates and publishes metrics
        """
        try:
            with self.monitoring_lock:
                # Calculate current metrics
                current_metrics = self.get_current_metrics()

                # Check for performance issues
                self.check_performance_thresholds(current_metrics)

                # Publish metrics
                self.publish_metrics(current_metrics)

                # Clean up old execution metrics
                self.cleanup_old_metrics()

        except Exception as e:
            self.get_logger().error(f'Error in monitoring callback: {str(e)}')

    def get_current_metrics(self) -> Dict[str, Any]:
        """
        Get current pipeline metrics
        """
        try:
            with self.monitoring_lock:
                # Calculate average metrics from history
                avg_execution_time = sum(self.metrics_history['execution_time']) / len(self.metrics_history['execution_time']) if self.metrics_history['execution_time'] else 0.0
                avg_success_rate = sum(self.metrics_history['success_rate']) / len(self.metrics_history['success_rate']) if self.metrics_history['success_rate'] else 0.0
                avg_error_rate = sum(self.metrics_history['error_rate']) / len(self.metrics_history['error_rate']) if self.metrics_history['error_rate'] else 0.0
                avg_throughput = sum(self.metrics_history['throughput']) / len(self.metrics_history['throughput']) if self.metrics_history['throughput'] else 0.0

                # Active execution count
                active_executions = sum(1 for metrics in self.execution_metrics.values()
                                      if metrics['current_status'] not in ['completed', 'failed', 'error'])

                metrics = {
                    'timestamp': time.time(),
                    'active_executions': active_executions,
                    'total_executions': len(self.pipeline_history),
                    'average_execution_time': avg_execution_time,
                    'average_success_rate': avg_success_rate,
                    'average_error_rate': avg_error_rate,
                    'average_throughput': avg_throughput,
                    'recent_execution_times': list(self.metrics_history['execution_time']),
                    'recent_success_rates': list(self.metrics_history['success_rate']),
                    'recent_error_rates': list(self.metrics_history['error_rate']),
                    'recent_throughput': list(self.metrics_history['throughput'])
                }

                return metrics

        except Exception as e:
            self.get_logger().error(f'Error getting current metrics: {str(e)}')
            return {}

    def check_performance_thresholds(self, metrics: Dict[str, Any]):
        """
        Check if performance metrics exceed thresholds
        """
        try:
            issues = []

            if metrics.get('average_execution_time', 0) > self.performance_thresholds['max_execution_time']:
                issues.append(f'Average execution time too high: {metrics["average_execution_time"]:.2f}s')

            if metrics.get('average_success_rate', 1.0) < self.performance_thresholds['min_success_rate']:
                issues.append(f'Success rate too low: {metrics["average_success_rate"]:.2f}')

            if metrics.get('average_error_rate', 0.0) > self.performance_thresholds['max_error_rate']:
                issues.append(f'Error rate too high: {metrics["average_error_rate"]:.2f}')

            if issues:
                self.get_logger().warn(f'Performance issues detected: {", ".join(issues)}')

        except Exception as e:
            self.get_logger().error(f'Error checking performance thresholds: {str(e)}')

    def publish_metrics(self, metrics: Dict[str, Any]):
        """
        Publish pipeline metrics
        """
        try:
            status_msg = ExecutionStatus()
            status_msg.header.stamp = self.get_clock().now().to_msg()
            status_msg.header.frame_id = 'pipeline_monitor'
            status_msg.execution_id = f'metrics_{int(time.time())}'
            status_msg.plan_id = 'pipeline_monitor'
            status_msg.overall_status = 'monitoring'
            status_msg.completed_tasks = len(metrics.get('recent_execution_times', []))
            status_msg.total_tasks = self.metrics_history_size
            status_msg.progress = 1.0
            status_msg.error = ''

            self.metrics_pub.publish(status_msg)

        except Exception as e:
            self.get_logger().error(f'Error publishing metrics: {str(e)}')

    def get_pipeline_performance(self) -> Dict[str, Any]:
        """
        Get comprehensive pipeline performance metrics
        """
        try:
            with self.monitoring_lock:
                metrics = self.get_current_metrics()

                # Additional performance metrics
                performance = {
                    'efficiency': self.calculate_efficiency(),
                    'reliability': self.calculate_reliability(),
                    'throughput': self.calculate_throughput(),
                    'latency': self.calculate_latency(),
                    'resource_utilization': self.get_resource_utilization()
                }

                metrics['performance'] = performance
                return metrics

        except Exception as e:
            self.get_logger().error(f'Error getting pipeline performance: {str(e)}')
            return {}

    def calculate_efficiency(self) -> float:
        """
        Calculate pipeline efficiency
        """
        try:
            if not self.pipeline_history:
                return 1.0  # Perfect efficiency if no history

            successful_executions = sum(1 for entry in self.pipeline_history if entry['success'])
            total_executions = len(self.pipeline_history)

            return successful_executions / total_executions if total_executions > 0 else 0.0

        except Exception as e:
            self.get_logger().error(f'Error calculating efficiency: {str(e)}')
            return 0.0

    def calculate_reliability(self) -> float:
        """
        Calculate pipeline reliability
        """
        try:
            if not self.pipeline_history:
                return 1.0  # Perfect reliability if no history

            non_error_executions = sum(1 for entry in self.pipeline_history if not entry['error'])
            total_executions = len(self.pipeline_history)

            return non_error_executions / total_executions if total_executions > 0 else 0.0

        except Exception as e:
            self.get_logger().error(f'Error calculating reliability: {str(e)}')
            return 0.0

    def calculate_throughput(self) -> float:
        """
        Calculate pipeline throughput (executions per minute)
        """
        try:
            if not self.pipeline_history:
                return 0.0

            if len(self.pipeline_history) < 2:
                return 0.0

            start_time = self.pipeline_history[0]['start_time']
            end_time = self.pipeline_history[-1]['end_time']
            time_span = end_time - start_time

            if time_span <= 0:
                return 0.0

            # Throughput in executions per minute
            throughput = (len(self.pipeline_history) / time_span) * 60
            return throughput

        except Exception as e:
            self.get_logger().error(f'Error calculating throughput: {str(e)}')
            return 0.0

    def calculate_latency(self) -> float:
        """
        Calculate average execution latency
        """
        try:
            if not self.pipeline_history:
                return 0.0

            total_latency = sum(entry['execution_time'] for entry in self.pipeline_history)
            return total_latency / len(self.pipeline_history) if self.pipeline_history else 0.0

        except Exception as e:
            self.get_logger().error(f'Error calculating latency: {str(e)}')
            return 0.0

    def get_resource_utilization(self) -> Dict[str, float]:
        """
        Get resource utilization metrics (placeholder implementation)
        """
        try:
            # In a real implementation, this would monitor actual resource usage
            # For simulation, we'll return placeholder values
            return {
                'cpu_usage': 0.0,
                'memory_usage': 0.0,
                'network_usage': 0.0,
                'disk_usage': 0.0
            }
        except Exception as e:
            self.get_logger().error(f'Error getting resource utilization: {str(e)}')
            return {}

    def get_component_status(self) -> Dict[str, Any]:
        """
        Get status of pipeline components
        """
        try:
            with self.monitoring_lock:
                return self.component_status.copy()
        except Exception as e:
            self.get_logger().error(f'Error getting component status: {str(e)}')
            return {}

    def log_pipeline_event(self, event_type: str, status_msg: ExecutionStatus):
        """
        Log pipeline events for analysis
        """
        try:
            event = {
                'timestamp': time.time(),
                'event_type': event_type,
                'execution_id': status_msg.execution_id,
                'status': status_msg.overall_status,
                'progress': status_msg.progress,
                'completed_tasks': status_msg.completed_tasks,
                'total_tasks': status_msg.total_tasks
            }

            # In a real implementation, this might write to a log file or database
            # For now, we'll just keep it in memory
            if 'pipeline_events' not in self.__dict__:
                self.pipeline_events: List[Dict[str, Any]] = []
            self.pipeline_events.append(event)

            # Limit event history
            if len(self.pipeline_events) > self.metrics_history_size * 10:
                self.pipeline_events = self.pipeline_events[-self.metrics_history_size * 10:]

        except Exception as e:
            self.get_logger().error(f'Error logging pipeline event: {str(e)}')

    def cleanup_old_metrics(self):
        """
        Clean up old execution metrics to prevent memory buildup
        """
        try:
            current_time = time.time()
            timeout = 600.0  # 10 minutes

            old_executions = []
            for exec_id, metrics in self.execution_metrics.items():
                if current_time - metrics['last_update'] > timeout:
                    old_executions.append(exec_id)

            for exec_id in old_executions:
                del self.execution_metrics[exec_id]

        except Exception as e:
            self.get_logger().error(f'Error cleaning up old metrics: {str(e)}')

    def reset_monitor(self):
        """
        Reset the monitor to initial state
        """
        try:
            with self.monitoring_lock:
                self.execution_metrics.clear()
                self.pipeline_history.clear()
                for key in self.metrics_history:
                    self.metrics_history[key].clear()
                self.component_status.clear()
                if hasattr(self, 'pipeline_events'):
                    delattr(self, 'pipeline_events')

            self.get_logger().info('Pipeline Monitor reset')

        except Exception as e:
            self.get_logger().error(f'Error resetting monitor: {str(e)}')

    def get_execution_analytics(self, execution_id: str) -> Dict[str, Any]:
        """
        Get detailed analytics for a specific execution
        """
        try:
            if execution_id not in self.execution_metrics:
                return {}

            metrics = self.execution_metrics[execution_id]
            analytics = {
                'execution_id': execution_id,
                'duration': time.time() - metrics['start_time'],
                'final_status': metrics['current_status'],
                'progress_over_time': metrics['progress_history'][:],  # Copy the list
                'status_changes': len(metrics['status_history']),
                'status_transitions': [entry['status'] for entry in metrics['status_history']],
                'tasks_completed': metrics['completed_tasks'],
                'tasks_total': metrics['total_tasks']
            }

            return analytics

        except Exception as e:
            self.get_logger().error(f'Error getting execution analytics: {str(e)}')
            return {}

    def get_trend_analysis(self) -> Dict[str, Any]:
        """
        Get trend analysis for pipeline performance
        """
        try:
            with self.monitoring_lock:
                trends = {
                    'execution_time_trend': self.calculate_trend(list(self.metrics_history['execution_time'])),
                    'success_rate_trend': self.calculate_trend(list(self.metrics_history['success_rate'])),
                    'error_rate_trend': self.calculate_trend(list(self.metrics_history['error_rate'])),
                    'throughput_trend': self.calculate_trend(list(self.metrics_history['throughput']))
                }

                return trends

        except Exception as e:
            self.get_logger().error(f'Error getting trend analysis: {str(e)}')
            return {}

    def calculate_trend(self, values: List[float]) -> str:
        """
        Calculate trend direction for a series of values
        """
        try:
            if len(values) < 2:
                return 'insufficient_data'

            # Simple trend analysis based on first and last values
            if len(values) >= 10:
                recent_avg = sum(values[-5:]) / 5
                earlier_avg = sum(values[:5]) / 5
            else:
                recent_avg = values[-1]
                earlier_avg = values[0]

            if recent_avg > earlier_avg * 1.1:
                return 'increasing'
            elif recent_avg < earlier_avg * 0.9:
                return 'decreasing'
            else:
                return 'stable'

        except Exception as e:
            self.get_logger().error(f'Error calculating trend: {str(e)}')
            return 'error'


def main(args=None):
    rclpy.init(args=args)

    pipeline_monitor = PipelineMonitorNode()

    try:
        rclpy.spin(pipeline_monitor)
    except KeyboardInterrupt:
        pass
    finally:
        pipeline_monitor.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()