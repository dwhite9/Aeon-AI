"""
Code Executor - Kubernetes Job-based code execution

Executes Python code in isolated Kubernetes Jobs with:
- Resource limits
- Timeout enforcement
- Output capture
- Security restrictions
"""

import asyncio
import base64
import hashlib
import logging
import time
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any
from dataclasses import dataclass

try:
    from kubernetes import client, config
    from kubernetes.client.rest import ApiException
    KUBERNETES_AVAILABLE = True
except ImportError:
    KUBERNETES_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Kubernetes client not available - code execution disabled")

from .validator import CodeValidator, ValidationResult

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """Status of code execution"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    VALIDATION_FAILED = "validation_failed"


@dataclass
class ExecutionResult:
    """Result of code execution"""
    status: ExecutionStatus
    output: str
    error: Optional[str] = None
    execution_time: float = 0.0
    job_name: Optional[str] = None
    validation: Optional[ValidationResult] = None


class CodeExecutor:
    """
    Executes Python code in Kubernetes Jobs

    Provides safe, isolated execution with resource limits and timeout.
    """

    def __init__(
        self,
        namespace: str = "default",
        executor_image: str = "python:3.11-slim",
        cpu_limit: str = "500m",
        memory_limit: str = "512Mi",
        timeout_seconds: int = 30,
        max_output_lines: int = 1000
    ):
        """
        Initialize code executor

        Args:
            namespace: Kubernetes namespace for Jobs
            executor_image: Docker image for code execution
            cpu_limit: CPU limit (e.g., "500m")
            memory_limit: Memory limit (e.g., "512Mi")
            timeout_seconds: Maximum execution time
            max_output_lines: Maximum output lines to capture
        """
        self.namespace = namespace
        self.executor_image = executor_image
        self.cpu_limit = cpu_limit
        self.memory_limit = memory_limit
        self.timeout_seconds = timeout_seconds
        self.max_output_lines = max_output_lines

        # Initialize validator
        self.validator = CodeValidator()

        # Initialize Kubernetes client
        if KUBERNETES_AVAILABLE:
            try:
                # Try to load in-cluster config first
                config.load_incluster_config()
                logger.info("Loaded in-cluster Kubernetes config")
            except:
                try:
                    # Fall back to kubeconfig
                    config.load_kube_config()
                    logger.info("Loaded Kubernetes config from kubeconfig")
                except:
                    logger.warning("Could not load Kubernetes config")
                    self.k8s_available = False
                    return

            self.batch_v1 = client.BatchV1Api()
            self.core_v1 = client.CoreV1Api()
            self.k8s_available = True
        else:
            self.k8s_available = False
            logger.warning("Kubernetes client not available")

    async def execute(self, code: str, description: str = "Code execution") -> ExecutionResult:
        """
        Execute Python code in a Kubernetes Job

        Args:
            code: Python code to execute
            description: Description of the execution (for logging)

        Returns:
            ExecutionResult with output, status, and metadata
        """
        start_time = time.time()

        # Validate code first
        validation = self.validator.validate(code)
        if not validation.is_valid:
            logger.warning(f"Code validation failed: {validation.errors}")
            return ExecutionResult(
                status=ExecutionStatus.VALIDATION_FAILED,
                output="",
                error="; ".join(validation.errors),
                execution_time=time.time() - start_time,
                validation=validation
            )

        # Check if Kubernetes is available
        if not self.k8s_available:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                output="",
                error="Kubernetes is not available for code execution",
                execution_time=time.time() - start_time,
                validation=validation
            )

        # Generate unique job name
        code_hash = hashlib.sha256(code.encode()).hexdigest()[:8]
        timestamp = int(time.time())
        job_name = f"code-exec-{timestamp}-{code_hash}"

        try:
            # Create and run the Job
            job = self._create_job(job_name, code, description)
            self.batch_v1.create_namespaced_job(
                namespace=self.namespace,
                body=job
            )

            logger.info(f"Created Job {job_name} for code execution")

            # Wait for completion with timeout
            result = await self._wait_for_job(job_name, start_time)
            return result

        except ApiException as e:
            logger.error(f"Kubernetes API error: {e}")
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                output="",
                error=f"Kubernetes error: {e.reason}",
                execution_time=time.time() - start_time,
                job_name=job_name,
                validation=validation
            )

        except Exception as e:
            logger.error(f"Code execution error: {e}", exc_info=True)
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                output="",
                error=str(e),
                execution_time=time.time() - start_time,
                job_name=job_name,
                validation=validation
            )

        finally:
            # Cleanup: delete the Job (with propagationPolicy to delete pods too)
            try:
                await asyncio.sleep(2)  # Give time for logs to be captured
                self.batch_v1.delete_namespaced_job(
                    name=job_name,
                    namespace=self.namespace,
                    propagation_policy='Background'
                )
                logger.info(f"Cleaned up Job {job_name}")
            except:
                pass  # Ignore cleanup errors

    def _create_job(self, job_name: str, code: str, description: str) -> client.V1Job:
        """Create Kubernetes Job spec for code execution"""

        # Encode code as base64 to avoid shell escaping issues
        code_b64 = base64.b64encode(code.encode()).decode()

        # Command to decode and execute code
        command = [
            "/bin/sh",
            "-c",
            f"echo '{code_b64}' | base64 -d | python3 -u"
        ]

        # Create container spec
        container = client.V1Container(
            name="executor",
            image=self.executor_image,
            command=command,
            resources=client.V1ResourceRequirements(
                limits={
                    "cpu": self.cpu_limit,
                    "memory": self.memory_limit
                },
                requests={
                    "cpu": "100m",
                    "memory": "128Mi"
                }
            ),
            security_context=client.V1SecurityContext(
                run_as_non_root=True,
                run_as_user=65534,  # nobody user
                allow_privilege_escalation=False,
                read_only_root_filesystem=False,  # Python needs to write to /tmp
                capabilities=client.V1Capabilities(
                    drop=["ALL"]
                )
            )
        )

        # Create pod template
        template = client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(
                labels={
                    "app": "code-executor",
                    "job-name": job_name
                },
                annotations={
                    "description": description
                }
            ),
            spec=client.V1PodSpec(
                containers=[container],
                restart_policy="Never",
                active_deadline_seconds=self.timeout_seconds,
                automount_service_account_token=False
            )
        )

        # Create Job spec
        job = client.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=client.V1ObjectMeta(
                name=job_name,
                labels={
                    "app": "code-executor",
                    "managed-by": "aeon"
                }
            ),
            spec=client.V1JobSpec(
                template=template,
                backoff_limit=0,  # Don't retry on failure
                ttl_seconds_after_finished=120  # Auto-cleanup after 2 minutes
            )
        )

        return job

    async def _wait_for_job(self, job_name: str, start_time: float) -> ExecutionResult:
        """
        Wait for Job to complete and capture output

        Args:
            job_name: Name of the Job
            start_time: Execution start time

        Returns:
            ExecutionResult with captured output
        """
        poll_interval = 0.5  # Poll every 500ms
        max_wait = self.timeout_seconds + 5  # Extra buffer

        while time.time() - start_time < max_wait:
            try:
                # Get Job status
                job = self.batch_v1.read_namespaced_job_status(
                    name=job_name,
                    namespace=self.namespace
                )

                # Check if completed
                if job.status.succeeded:
                    # Get pod logs
                    output = await self._get_pod_logs(job_name)
                    return ExecutionResult(
                        status=ExecutionStatus.COMPLETED,
                        output=output,
                        execution_time=time.time() - start_time,
                        job_name=job_name
                    )

                elif job.status.failed:
                    # Get pod logs for error
                    output = await self._get_pod_logs(job_name)
                    error = self._extract_error(output)
                    return ExecutionResult(
                        status=ExecutionStatus.FAILED,
                        output=output,
                        error=error,
                        execution_time=time.time() - start_time,
                        job_name=job_name
                    )

                # Still running, wait
                await asyncio.sleep(poll_interval)

            except ApiException as e:
                if e.status == 404:
                    # Job not found yet, wait
                    await asyncio.sleep(poll_interval)
                else:
                    raise

        # Timeout
        return ExecutionResult(
            status=ExecutionStatus.TIMEOUT,
            output="",
            error=f"Execution timeout after {self.timeout_seconds} seconds",
            execution_time=time.time() - start_time,
            job_name=job_name
        )

    async def _get_pod_logs(self, job_name: str) -> str:
        """Get logs from the Job's pod"""
        try:
            # Find pod for this Job
            pods = self.core_v1.list_namespaced_pod(
                namespace=self.namespace,
                label_selector=f"job-name={job_name}"
            )

            if not pods.items:
                return ""

            pod_name = pods.items[0].metadata.name

            # Get logs
            logs = self.core_v1.read_namespaced_pod_log(
                name=pod_name,
                namespace=self.namespace,
                tail_lines=self.max_output_lines
            )

            return logs

        except Exception as e:
            logger.error(f"Error getting pod logs: {e}")
            return f"Error retrieving logs: {str(e)}"

    def _extract_error(self, output: str) -> str:
        """Extract error message from output"""
        lines = output.split('\n')
        # Look for Python tracebacks
        for i, line in enumerate(lines):
            if 'Traceback' in line:
                # Return everything from traceback onwards
                return '\n'.join(lines[i:])
        # If no traceback, return last non-empty line
        for line in reversed(lines):
            if line.strip():
                return line
        return "Unknown error"

    def get_safe_modules(self) -> list:
        """Get list of safe modules for code execution"""
        return self.validator.get_safe_imports()
