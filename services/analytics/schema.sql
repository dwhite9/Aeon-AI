-- Analytics Database Schema
-- Creates tables for query tracking, metrics aggregation, and optimization

-- Hourly aggregated metrics
CREATE TABLE IF NOT EXISTS hourly_metrics (
    id SERIAL PRIMARY KEY,
    hour TIMESTAMP NOT NULL,
    total_queries INTEGER NOT NULL DEFAULT 0,
    successful_queries INTEGER NOT NULL DEFAULT 0,
    avg_execution_time FLOAT NOT NULL DEFAULT 0,
    cache_hit_rate FLOAT NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(hour)
);

CREATE INDEX idx_hourly_metrics_hour ON hourly_metrics(hour DESC);

-- Tool usage statistics
CREATE TABLE IF NOT EXISTS tool_usage_stats (
    id SERIAL PRIMARY KEY,
    hour TIMESTAMP NOT NULL,
    tool_name VARCHAR(50) NOT NULL,
    total_calls INTEGER NOT NULL DEFAULT 0,
    successful_calls INTEGER NOT NULL DEFAULT 0,
    avg_execution_time FLOAT NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(hour, tool_name)
);

CREATE INDEX idx_tool_usage_hour ON tool_usage_stats(hour DESC);
CREATE INDEX idx_tool_usage_tool ON tool_usage_stats(tool_name);

-- Query logs table (extends the one from rag/analytics.py)
-- This ensures the table exists with all needed columns
CREATE TABLE IF NOT EXISTS query_logs (
    id SERIAL PRIMARY KEY,
    query_id VARCHAR(64) UNIQUE NOT NULL,
    query_text TEXT,
    tool_used VARCHAR(50),
    execution_time FLOAT,
    success BOOLEAN DEFAULT TRUE,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    session_id VARCHAR(64),
    error_message TEXT,
    cache_hit BOOLEAN DEFAULT FALSE,
    result_quality FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_query_logs_timestamp ON query_logs(timestamp DESC);
CREATE INDEX idx_query_logs_session ON query_logs(session_id);
CREATE INDEX idx_query_logs_tool ON query_logs(tool_used);
CREATE INDEX idx_query_logs_success ON query_logs(success);

-- Optimization reports
CREATE TABLE IF NOT EXISTS optimization_reports (
    id SERIAL PRIMARY KEY,
    generated_at TIMESTAMP NOT NULL,
    status VARCHAR(20) NOT NULL, -- 'healthy', 'moderate', 'high', 'critical'
    total_recommendations INTEGER NOT NULL DEFAULT 0,
    critical_count INTEGER NOT NULL DEFAULT 0,
    high_count INTEGER NOT NULL DEFAULT 0,
    medium_count INTEGER NOT NULL DEFAULT 0,
    low_count INTEGER NOT NULL DEFAULT 0,
    report_data JSONB, -- Full report in JSON format
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_optimization_reports_generated ON optimization_reports(generated_at DESC);

-- Fine-tuning training data
CREATE TABLE IF NOT EXISTS training_examples (
    id SERIAL PRIMARY KEY,
    query TEXT NOT NULL,
    positive_documents JSONB,
    negative_documents JSONB,
    relevance_score FLOAT,
    source VARCHAR(50), -- 'user_feedback', 'implicit', 'synthetic'
    timestamp TIMESTAMP NOT NULL,
    used_in_training BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_training_examples_timestamp ON training_examples(timestamp DESC);
CREATE INDEX idx_training_examples_used ON training_examples(used_in_training);

-- Fine-tuning runs
CREATE TABLE IF NOT EXISTS finetuning_runs (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(64) UNIQUE NOT NULL,
    started_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP,
    status VARCHAR(20) NOT NULL, -- 'running', 'completed', 'failed'
    base_model VARCHAR(100) NOT NULL,
    output_model_path TEXT,
    training_examples_count INTEGER,
    epochs INTEGER,
    batch_size INTEGER,
    learning_rate FLOAT,
    evaluation_metrics JSONB,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_finetuning_runs_started ON finetuning_runs(started_at DESC);
CREATE INDEX idx_finetuning_runs_status ON finetuning_runs(status);

-- System health snapshots
CREATE TABLE IF NOT EXISTS health_snapshots (
    id SERIAL PRIMARY KEY,
    snapshot_time TIMESTAMP NOT NULL,
    overall_status VARCHAR(20) NOT NULL, -- 'healthy', 'degraded', 'unhealthy'
    redis_status VARCHAR(20),
    qdrant_status VARCHAR(20),
    postgres_status VARCHAR(20),
    vllm_status VARCHAR(20),
    embedding_status VARCHAR(20),
    agent_status VARCHAR(20),
    metrics JSONB, -- Additional metrics
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_health_snapshots_time ON health_snapshots(snapshot_time DESC);

-- Comments for documentation
COMMENT ON TABLE hourly_metrics IS 'Aggregated query metrics by hour';
COMMENT ON TABLE tool_usage_stats IS 'Tool usage statistics aggregated by hour';
COMMENT ON TABLE query_logs IS 'Individual query execution logs';
COMMENT ON TABLE optimization_reports IS 'Generated optimization recommendations';
COMMENT ON TABLE training_examples IS 'Training data for embedding fine-tuning';
COMMENT ON TABLE finetuning_runs IS 'History of fine-tuning runs';
COMMENT ON TABLE health_snapshots IS 'Periodic system health checks';
