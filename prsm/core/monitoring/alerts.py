"""
PRSM Alert Management System
============================

Comprehensive alerting system for PRSM monitoring.
Handles alert rules, notifications, and escalation policies.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod

try:
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    FATAL = "fatal"


class AlertStatus(Enum):
    """Alert status"""
    PENDING = "pending"
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class AlertCondition:
    """Definition of an alert condition"""
    metric_name: str
    operator: str  # gt, lt, eq, ne, gte, lte
    threshold: float
    duration: timedelta = field(default_factory=lambda: timedelta(minutes=5))
    aggregation: str = "avg"  # avg, max, min, sum, count


@dataclass
class Alert:
    """An alert instance"""
    id: str
    rule_name: str
    severity: AlertSeverity
    status: AlertStatus
    message: str
    created_at: datetime
    updated_at: datetime
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    escalation_level: int = 0


class AlertChannel(ABC):
    """Base class for alert notification channels"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.enabled = config.get("enabled", True)
    
    @abstractmethod
    async def send_alert(self, alert: Alert) -> bool:
        """Send alert notification"""
        pass
    
    def format_message(self, alert: Alert) -> str:
        """Format alert message"""
        return f"""PRSM Alert: {alert.rule_name}

Severity: {alert.severity.value.upper()}
Status: {alert.status.value.upper()}
Message: {alert.message}
Created: {alert.created_at.isoformat()}

Alert ID: {alert.id}
"""


class EmailAlertChannel(AlertChannel):
    """Email alert channel"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.smtp_server = config.get("smtp_server", "localhost")
        self.smtp_port = config.get("smtp_port", 587)
        self.username = config.get("username")
        self.password = config.get("password")
        self.from_email = config.get("from_email", "alerts@prsm.local")
        self.to_emails = config.get("to_emails", [])
        self.use_tls = config.get("use_tls", True)
    
    async def send_alert(self, alert: Alert) -> bool:
        """Send email alert"""
        if not EMAIL_AVAILABLE:
            logger.error("Email libraries not available")
            return False
        
        if not self.to_emails:
            logger.error("No recipient emails configured")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = ", ".join(self.to_emails)
            msg['Subject'] = f"PRSM Alert: {alert.rule_name} [{alert.severity.value.upper()}]"
            
            # Add body
            body = self.format_message(alert)
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            
            if self.use_tls:
                server.starttls()
            
            if self.username and self.password:
                server.login(self.username, self.password)
            
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email alert sent for {alert.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False


class WebhookAlertChannel(AlertChannel):
    """Webhook alert channel"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.url = config.get("url")
        self.method = config.get("method", "POST")
        self.headers = config.get("headers", {})
        self.timeout = config.get("timeout", 10)
    
    async def send_alert(self, alert: Alert) -> bool:
        """Send webhook alert"""
        if not REQUESTS_AVAILABLE:
            logger.error("Requests library not available")
            return False
        
        if not self.url:
            logger.error("Webhook URL not configured")
            return False
        
        try:
            # Prepare payload
            payload = {
                "alert_id": alert.id,
                "rule_name": alert.rule_name,
                "severity": alert.severity.value,
                "status": alert.status.value,
                "message": alert.message,
                "created_at": alert.created_at.isoformat(),
                "metadata": alert.metadata
            }
            
            # Send webhook
            response = requests.request(
                method=self.method,
                url=self.url,
                json=payload,
                headers=self.headers,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            
            logger.info(f"Webhook alert sent for {alert.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            return False


class SlackAlertChannel(AlertChannel):
    """Slack alert channel"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.webhook_url = config.get("webhook_url")
        self.channel = config.get("channel", "#alerts")
        self.username = config.get("username", "PRSM Alerts")
        self.icon_emoji = config.get("icon_emoji", ":warning:")
    
    async def send_alert(self, alert: Alert) -> bool:
        """Send Slack alert"""
        if not REQUESTS_AVAILABLE:
            logger.error("Requests library not available")
            return False
        
        if not self.webhook_url:
            logger.error("Slack webhook URL not configured")
            return False
        
        try:
            # Color based on severity
            color_map = {
                AlertSeverity.INFO: "good",
                AlertSeverity.WARNING: "warning", 
                AlertSeverity.CRITICAL: "danger",
                AlertSeverity.FATAL: "danger"
            }
            
            # Prepare Slack payload
            payload = {
                "channel": self.channel,
                "username": self.username,
                "icon_emoji": self.icon_emoji,
                "attachments": [{
                    "color": color_map.get(alert.severity, "warning"),
                    "title": f"PRSM Alert: {alert.rule_name}",
                    "text": alert.message,
                    "fields": [
                        {
                            "title": "Severity",
                            "value": alert.severity.value.upper(),
                            "short": True
                        },
                        {
                            "title": "Status",
                            "value": alert.status.value.upper(),
                            "short": True
                        },
                        {
                            "title": "Alert ID",
                            "value": alert.id,
                            "short": True
                        },
                        {
                            "title": "Created",
                            "value": alert.created_at.strftime("%Y-%m-%d %H:%M:%S UTC"),
                            "short": True
                        }
                    ],
                    "timestamp": int(alert.created_at.timestamp())
                }]
            }
            
            # Send to Slack
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )
            
            response.raise_for_status()
            
            logger.info(f"Slack alert sent for {alert.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False


class AlertRule:
    """Alert rule definition"""
    
    def __init__(self, name: str, description: str, condition: AlertCondition, 
                 severity: AlertSeverity, channels: List[str], 
                 enabled: bool = True, tags: Optional[Dict[str, str]] = None):
        self.name = name
        self.description = description
        self.condition = condition
        self.severity = severity
        self.channels = channels
        self.enabled = enabled
        self.tags = tags or {}
        self.created_at = datetime.now()
        self.last_triggered = None
        self.trigger_count = 0
        self._condition_start_time = None
        self._condition_values = []
    
    def evaluate(self, metric_value: float, timestamp: datetime) -> bool:
        """Evaluate if alert condition is met"""
        if not self.enabled:
            return False
        
        # Check if condition is met
        condition_met = self._check_condition(metric_value)
        
        if condition_met:
            if self._condition_start_time is None:
                self._condition_start_time = timestamp
                self._condition_values = [metric_value]
            else:
                self._condition_values.append(metric_value)
                
                # Check if condition has been met for required duration
                if timestamp - self._condition_start_time >= self.condition.duration:
                    # Calculate aggregated value
                    if self.condition.aggregation == "avg":
                        agg_value = sum(self._condition_values) / len(self._condition_values)
                    elif self.condition.aggregation == "max":
                        agg_value = max(self._condition_values)
                    elif self.condition.aggregation == "min":
                        agg_value = min(self._condition_values)
                    elif self.condition.aggregation == "sum":
                        agg_value = sum(self._condition_values)
                    else:  # count
                        agg_value = len(self._condition_values)
                    
                    # Check if aggregated value still meets condition
                    if self._check_condition(agg_value):
                        self.trigger_count += 1
                        self.last_triggered = timestamp
                        self._reset_condition_tracking()
                        return True
        else:
            self._reset_condition_tracking()
        
        return False
    
    def _check_condition(self, value: float) -> bool:
        """Check if value meets condition"""
        if self.condition.operator == "gt":
            return value > self.condition.threshold
        elif self.condition.operator == "lt":
            return value < self.condition.threshold
        elif self.condition.operator == "eq":
            return value == self.condition.threshold
        elif self.condition.operator == "ne":
            return value != self.condition.threshold
        elif self.condition.operator == "gte":
            return value >= self.condition.threshold
        elif self.condition.operator == "lte":
            return value <= self.condition.threshold
        else:
            return False
    
    def _reset_condition_tracking(self):
        """Reset condition tracking state"""
        self._condition_start_time = None
        self._condition_values = []


class AlertManager:
    """Main alert management system"""
    
    def __init__(self, evaluation_interval: float = 30.0):
        self.evaluation_interval = evaluation_interval
        self.rules: Dict[str, AlertRule] = {}
        self.channels: Dict[str, AlertChannel] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.is_monitoring = False
        self._monitoring_task: Optional[asyncio.Task] = None
        self._metrics_collector = None
        self.max_history = 10000
    
    def set_metrics_collector(self, metrics_collector):
        """Set metrics collector for evaluation"""
        self._metrics_collector = metrics_collector
    
    def add_rule(self, rule: AlertRule) -> None:
        """Add alert rule"""
        self.rules[rule.name] = rule
        logger.info(f"Added alert rule: {rule.name}")
    
    def remove_rule(self, name: str) -> None:
        """Remove alert rule"""
        if name in self.rules:
            del self.rules[name]
            logger.info(f"Removed alert rule: {name}")
    
    def add_channel(self, channel: AlertChannel) -> None:
        """Add alert channel"""
        self.channels[channel.name] = channel
        logger.info(f"Added alert channel: {channel.name}")
    
    def remove_channel(self, name: str) -> None:
        """Remove alert channel"""
        if name in self.channels:
            del self.channels[name]
            logger.info(f"Removed alert channel: {name}")
    
    async def create_alert(self, rule: AlertRule, metric_value: float) -> Alert:
        """Create new alert"""
        alert_id = f"{rule.name}_{int(datetime.now().timestamp())}"
        
        alert = Alert(
            id=alert_id,
            rule_name=rule.name,
            severity=rule.severity,
            status=AlertStatus.ACTIVE,
            message=f"Alert triggered: {rule.description}. Current value: {metric_value}, threshold: {rule.condition.threshold}",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata={
                "metric_name": rule.condition.metric_name,
                "metric_value": metric_value,
                "threshold": rule.condition.threshold,
                "operator": rule.condition.operator,
                "rule_tags": rule.tags
            }
        )
        
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Trim history if needed
        if len(self.alert_history) > self.max_history:
            self.alert_history = self.alert_history[-self.max_history:]
        
        # Send notifications
        await self._send_notifications(alert, rule)
        
        logger.warning(f"Alert created: {alert_id} - {alert.message}")
        return alert
    
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert"""
        if alert_id not in self.active_alerts:
            return False
        
        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_at = datetime.now()
        alert.acknowledged_by = acknowledged_by
        alert.updated_at = datetime.now()
        
        logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
        return True
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        if alert_id not in self.active_alerts:
            return False
        
        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.now()
        alert.updated_at = datetime.now()
        
        # Remove from active alerts
        del self.active_alerts[alert_id]
        
        logger.info(f"Alert resolved: {alert_id}")
        return True
    
    async def _send_notifications(self, alert: Alert, rule: AlertRule) -> None:
        """Send alert notifications to configured channels"""
        for channel_name in rule.channels:
            if channel_name not in self.channels:
                logger.error(f"Alert channel not found: {channel_name}")
                continue
            
            channel = self.channels[channel_name]
            if not channel.enabled:
                continue
            
            try:
                success = await channel.send_alert(alert)
                if success:
                    logger.info(f"Alert sent via {channel_name}: {alert.id}")
                else:
                    logger.error(f"Failed to send alert via {channel_name}: {alert.id}")
            except Exception as e:
                logger.error(f"Error sending alert via {channel_name}: {e}")
    
    async def start_monitoring(self) -> None:
        """Start alert monitoring"""
        if self.is_monitoring:
            logger.warning("Alert monitoring already started")
            return
        
        if not self._metrics_collector:
            logger.error("No metrics collector configured")
            return
        
        self.is_monitoring = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info(f"Started alert monitoring with {self.evaluation_interval}s interval")
    
    async def stop_monitoring(self) -> None:
        """Stop alert monitoring"""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped alert monitoring")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                await self._evaluate_rules()
                
            except Exception as e:
                logger.error(f"Error in alert monitoring loop: {e}")
            
            try:
                await asyncio.sleep(self.evaluation_interval)
            except asyncio.CancelledError:
                break
    
    async def _evaluate_rules(self) -> None:
        """Evaluate all alert rules"""
        if not self._metrics_collector:
            return
        
        # Get current metrics
        current_metrics = await self._metrics_collector.registry.collect_all()
        
        # Group metrics by name
        metrics_by_name = {}
        for metric in current_metrics:
            if metric.name not in metrics_by_name:
                metrics_by_name[metric.name] = []
            metrics_by_name[metric.name].append(metric)
        
        # Evaluate each rule
        for rule in self.rules.values():
            if not rule.enabled:
                continue
            
            # Find matching metrics
            if rule.condition.metric_name not in metrics_by_name:
                continue
            
            # Get the most recent metric value
            metric_values = metrics_by_name[rule.condition.metric_name]
            if not metric_values:
                continue
            
            # Sort by timestamp and get the latest
            metric_values.sort(key=lambda m: m.timestamp, reverse=True)
            latest_metric = metric_values[0]
            
            # Evaluate rule
            try:
                should_alert = rule.evaluate(float(latest_metric.value), latest_metric.timestamp)
                
                if should_alert:
                    # Check if we already have an active alert for this rule
                    existing_alert = None
                    for alert in self.active_alerts.values():
                        if alert.rule_name == rule.name and alert.status == AlertStatus.ACTIVE:
                            existing_alert = alert
                            break
                    
                    if not existing_alert:
                        await self.create_alert(rule, float(latest_metric.value))
                    
            except Exception as e:
                logger.error(f"Error evaluating rule {rule.name}: {e}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alert_history if alert.created_at >= cutoff_time]
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary statistics"""
        active_by_severity = {severity: 0 for severity in AlertSeverity}
        for alert in self.active_alerts.values():
            active_by_severity[alert.severity] += 1
        
        recent_history = self.get_alert_history(24)
        
        return {
            "active_alerts_count": len(self.active_alerts),
            "active_by_severity": {k.value: v for k, v in active_by_severity.items()},
            "total_rules": len(self.rules),
            "enabled_rules": len([r for r in self.rules.values() if r.enabled]),
            "total_channels": len(self.channels),
            "enabled_channels": len([c for c in self.channels.values() if c.enabled]),
            "alerts_24h": len(recent_history),
            "monitoring_active": self.is_monitoring,
            "evaluation_interval": self.evaluation_interval
        }
    
    def setup_default_rules(self) -> None:
        """Setup default alert rules"""
        logger.info("Setting up default alert rules")
        
        # High error rate alert
        error_rule = AlertRule(
            name="high_error_rate",
            description="High error rate detected",
            condition=AlertCondition(
                metric_name="prsm_errors_total",
                operator="gt",
                threshold=10,
                duration=timedelta(minutes=5),
                aggregation="sum"
            ),
            severity=AlertSeverity.WARNING,
            channels=["email", "slack"],
            tags={"category": "system", "component": "core"}
        )
        
        # High response time alert
        response_time_rule = AlertRule(
            name="high_response_time",
            description="Query response time is too high",
            condition=AlertCondition(
                metric_name="prsm_query_duration_seconds",
                operator="gt",
                threshold=30.0,
                duration=timedelta(minutes=3),
                aggregation="avg"
            ),
            severity=AlertSeverity.CRITICAL,
            channels=["email", "slack", "webhook"],
            tags={"category": "performance", "component": "query"}
        )
        
        # Low quality score alert
        quality_rule = AlertRule(
            name="low_quality_score",
            description="Response quality score is below threshold",
            condition=AlertCondition(
                metric_name="prsm_quality_score",
                operator="lt",
                threshold=70.0,
                duration=timedelta(minutes=10),
                aggregation="avg"
            ),
            severity=AlertSeverity.WARNING,
            channels=["email"],
            tags={"category": "quality", "component": "response"}
        )
        
        # High FTNS usage alert
        ftns_rule = AlertRule(
            name="high_ftns_usage",
            description="FTNS usage is unusually high",
            condition=AlertCondition(
                metric_name="prsm_ftns_used",
                operator="gt",
                threshold=1000.0,
                duration=timedelta(minutes=15),
                aggregation="sum"
            ),
            severity=AlertSeverity.INFO,
            channels=["email"],
            tags={"category": "cost", "component": "ftns"}
        )
        
        # Add rules
        for rule in [error_rule, response_time_rule, quality_rule, ftns_rule]:
            self.add_rule(rule)

        logger.info(f"Setup {len(self.rules)} default alert rules")

    def setup_node_runtime_rules(self, channels: Optional[List[str]] = None) -> None:
        """Sprint 1217 — alert rules over a live PRSM node's runtime metrics
        (emitted by ``NodeRuntimeMetrics``).

        ``channels`` (sp1218): the channel names a fired rule notifies. Default
        None → ``[]`` so the rules register + log + populate ``active_alerts``
        WITHOUT requiring an email/slack/webhook channel (an operator scrapes
        ``active_alerts`` / logs). Pass e.g. ``["node-webhook"]`` (with a
        matching ``add_channel``) to deliver notifications."""
        chans: List[str] = list(channels) if channels else []
        logger.info("Setting up node-runtime alert rules (channels=%s)", chans)

        # Node not ready (can't serve the primary inference path) — fire fast.
        not_ready_rule = AlertRule(
            name="node_not_ready",
            description="Node is not ready to serve inference (primary path down)",
            condition=AlertCondition(
                metric_name="prsm_node_ready",
                operator="lt",
                threshold=1,
                duration=timedelta(minutes=1),
                aggregation="min",
            ),
            severity=AlertSeverity.CRITICAL,
            channels=list(chans),
            tags={"category": "availability", "component": "inference"},
        )

        # On-chain settlement funds continuously in-flight — a transient
        # in-flight during a normal commit is fine; SUSTAINED (min>0 over the
        # window) means a quarantined/unconfirmed commit may be stuck.
        funds_in_flight_rule = AlertRule(
            name="settlement_funds_in_flight_stuck",
            description="On-chain settlement funds in-flight for a sustained period",
            condition=AlertCondition(
                metric_name="prsm_settlement_funds_in_flight",
                operator="gt",
                threshold=0,
                duration=timedelta(minutes=15),
                aggregation="min",
            ),
            severity=AlertSeverity.WARNING,
            channels=list(chans),
            tags={"category": "settlement", "component": "escrow"},
        )

        # Commit-intent WAL entries unresolved — a sustained non-zero means a
        # commit may have landed on chain but its local promote was lost
        # (awaiting chain-scan recovery).
        committing_intents_rule = AlertRule(
            name="settlement_committing_intents_unresolved",
            description="Settlement commit-intent WAL entries unresolved (possible stranded commit)",
            condition=AlertCondition(
                metric_name="prsm_settlement_committing_intents",
                operator="gt",
                threshold=0,
                duration=timedelta(minutes=15),
                aggregation="min",
            ),
            severity=AlertSeverity.WARNING,
            channels=list(chans),
            tags={"category": "settlement", "component": "wal"},
        )

        for rule in [not_ready_rule, funds_in_flight_rule, committing_intents_rule]:
            self.add_rule(rule)
        logger.info("Setup %d node-runtime alert rules", 3)

    def setup_collaboration_rules(self) -> None:
        """Setup collaboration/trust-path alert rules for P3 observability.

        These rules are intentionally focused on bounded telemetry surfaces emitted
        by the collaboration observability bridge metrics.
        """
        logger.info("Setting up collaboration/trust alert rules")

        # Conservative production defaults:
        # - avoid noise on transient spikes,
        # - still escalate quickly on likely trust/reliability regressions.
        handshake_failure_rate_warning = 0.15
        replay_nonce_spike_critical = 5
        transport_dispatch_failure_rate_warning = 0.10
        manager_dispatch_failure_rate_warning = 0.08
        stalled_protocol_total_warning = 2
        stalled_protocol_ratio_warning = 0.25

        rules = [
            AlertRule(
                name="collab_handshake_failure_rate_high",
                description="Transport handshake failure rate is elevated",
                condition=AlertCondition(
                    metric_name="collab_transport_handshake_failure_rate",
                    operator="gt",
                    threshold=handshake_failure_rate_warning,
                    duration=timedelta(minutes=5),
                    aggregation="avg",
                ),
                severity=AlertSeverity.WARNING,
                channels=["email", "slack"],
                tags={"category": "trust", "component": "transport"},
            ),
            AlertRule(
                name="collab_replay_nonce_spike",
                description="Replay nonce rejection spike detected",
                condition=AlertCondition(
                    metric_name="collab_transport_handshake_replay_nonce_delta",
                    operator="gte",
                    threshold=replay_nonce_spike_critical,
                    duration=timedelta(minutes=2),
                    aggregation="max",
                ),
                severity=AlertSeverity.CRITICAL,
                channels=["email", "slack", "webhook"],
                tags={"category": "trust", "component": "transport"},
            ),
            AlertRule(
                name="collab_dispatch_failure_rate_high",
                description="Transport dispatch failure rate is elevated",
                condition=AlertCondition(
                    metric_name="collab_transport_dispatch_failure_rate",
                    operator="gt",
                    threshold=transport_dispatch_failure_rate_warning,
                    duration=timedelta(minutes=5),
                    aggregation="avg",
                ),
                severity=AlertSeverity.WARNING,
                channels=["email", "slack"],
                tags={"category": "collaboration", "component": "transport"},
            ),
            AlertRule(
                name="collab_manager_dispatch_failure_rate_high",
                description="Collaboration manager dispatch failure rate is elevated",
                condition=AlertCondition(
                    metric_name="collab_manager_dispatch_failure_rate",
                    operator="gt",
                    threshold=manager_dispatch_failure_rate_warning,
                    duration=timedelta(minutes=5),
                    aggregation="avg",
                ),
                severity=AlertSeverity.WARNING,
                channels=["email", "slack"],
                tags={"category": "collaboration", "component": "manager"},
            ),
            AlertRule(
                name="collab_stalled_protocols_detected",
                description="Collaboration protocols appear stalled (no terminal outcomes)",
                condition=AlertCondition(
                    metric_name="collab_protocol_stalled_total",
                    operator="gt",
                    threshold=stalled_protocol_total_warning,
                    duration=timedelta(minutes=5),
                    aggregation="max",
                ),
                severity=AlertSeverity.WARNING,
                channels=["email", "slack"],
                tags={"category": "collaboration", "component": "protocols"},
            ),
            AlertRule(
                name="collab_stalled_protocol_ratio_high",
                description="High ratio of collaboration protocols appears stalled",
                condition=AlertCondition(
                    metric_name="collab_protocol_stalled_ratio",
                    operator="gt",
                    threshold=stalled_protocol_ratio_warning,
                    duration=timedelta(minutes=5),
                    aggregation="avg",
                ),
                severity=AlertSeverity.WARNING,
                channels=["email", "slack"],
                tags={"category": "collaboration", "component": "protocols"},
            ),
        ]

        for rule in rules:
            self.add_rule(rule)

        logger.info("Setup %d collaboration/trust alert rules", len(rules))
    
    def setup_default_channels(self) -> None:
        """Setup default alert channels"""
        logger.info("Setting up default alert channels")
        
        # Email channel
        email_channel = EmailAlertChannel(
            "email",
            {
                "enabled": True,
                "smtp_server": "localhost",
                "smtp_port": 587,
                "from_email": "alerts@prsm.local",
                "to_emails": ["admin@prsm.local"],
                "use_tls": True
            }
        )
        
        # Slack channel
        slack_channel = SlackAlertChannel(
            "slack",
            {
                "enabled": False,  # Disabled by default, requires webhook URL
                "webhook_url": "",  # Configure in production
                "channel": "#prsm-alerts",
                "username": "PRSM Alerts",
                "icon_emoji": ":warning:"
            }
        )
        
        # Webhook channel
        webhook_channel = WebhookAlertChannel(
            "webhook",
            {
                "enabled": False,  # Disabled by default, requires URL
                "url": "",  # Configure in production
                "method": "POST",
                "headers": {"Content-Type": "application/json"},
                "timeout": 10
            }
        )
        
        # Add channels
        for channel in [email_channel, slack_channel, webhook_channel]:
            self.add_channel(channel)
        
        logger.info(f"Setup {len(self.channels)} default alert channels")
