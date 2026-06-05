-- Schema para feature store
CREATE SCHEMA IF NOT EXISTS feature_store;

-- Tabela de eventos (simulando logs do agente WASM)
CREATE TABLE IF NOT EXISTS feature_store.raw_events (
    event_id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    agent_id VARCHAR(50),
    event_type VARCHAR(50),
    latency_ms INTEGER,
    memory_usage_mb FLOAT,
    cpu_percent FLOAT,
    grok_tokens_used INTEGER,
    error_code VARCHAR(20),
    user_action VARCHAR(100),
    response_size_bytes INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- View v1: Features básicas (baseline)
CREATE OR REPLACE VIEW feature_store.vw_features_v1 AS
SELECT
    event_id,
    timestamp,
    agent_id,
    latency_ms,
    memory_usage_mb,
    cpu_percent,
    grok_tokens_used,
    error_code,
    user_action,
    response_size_bytes,
    -- Features derivadas
    CASE WHEN error_code IS NOT NULL THEN 1 ELSE 0 END AS has_error,
    CASE WHEN latency_ms > 1000 THEN 1 ELSE 0 END AS high_latency,
    EXTRACT(HOUR FROM timestamp) AS hour_of_day,
    EXTRACT(DOW FROM timestamp) AS day_of_week
FROM feature_store.raw_events;

-- View v2: Features avançadas (para drift detection)
CREATE OR REPLACE VIEW feature_store.vw_features_v2 AS
SELECT
    *,
    -- Janelas móveis
    LAG(latency_ms, 1) OVER (ORDER BY timestamp) AS prev_latency,
    AVG(latency_ms) OVER (PARTITION BY agent_id ORDER BY timestamp ROWS 5 PRECEDING) AS avg_latency_5,
    COUNT(*) OVER (PARTITION BY agent_id ORDER BY timestamp ROWS 10 PRECEDING) AS request_count_10,
    -- Categorização
    CASE 
        WHEN latency_ms < 200 THEN 'fast'
        WHEN latency_ms < 500 THEN 'normal'
        WHEN latency_ms < 1000 THEN 'slow'
        ELSE 'critical'
    END AS latency_category
FROM feature_store.vw_features_v1;

-- Índices para performance
CREATE INDEX IF NOT EXISTS idx_raw_events_timestamp ON feature_store.raw_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_raw_events_agent ON feature_store.raw_events(agent_id);
CREATE INDEX IF NOT EXISTS idx_raw_events_type ON feature_store.raw_events(event_type);

-- Tabela de referência para drift detection
CREATE TABLE IF NOT EXISTS feature_store.reference_data (
    ref_id SERIAL PRIMARY KEY,
    feature_name VARCHAR(100),
    mean_value FLOAT,
    std_value FLOAT,
    min_value FLOAT,
    max_value FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Dados de exemplo (baseline normal)
INSERT INTO feature_store.raw_events 
    (agent_id, event_type, latency_ms, memory_usage_mb, cpu_percent, 
     grok_tokens_used, error_code, user_action, response_size_bytes)
VALUES
    ('agent-001', 'success', 245, 512.5, 15.2, 150, NULL, 'chat', 1024),
    ('agent-001', 'success', 189, 520.1, 12.8, 200, NULL, 'chat', 2048),
    ('agent-002', 'success', 567, 480.0, 22.5, 350, NULL, 'file_upload', 5120),
    ('agent-001', 'error', 1200, 600.0, 45.0, 50, 'TIMEOUT', 'tool_call', 0),
    ('agent-003', 'success', 123, 410.2, 8.5, 80, NULL, 'chat', 512),
    ('agent-002', 'success', 334, 495.3, 18.1, 220, NULL, 'chat', 1536),
    ('agent-001', 'success', 278, 530.7, 16.0, 180, NULL, 'chat', 1024),
    ('agent-003', 'error', 890, 550.0, 35.0, 100, 'RATE_LIMIT', 'tool_call', 0),
    ('agent-002', 'success', 445, 510.0, 20.0, 280, NULL, 'file_upload', 3072),
    ('agent-001', 'success', 201, 505.5, 14.5, 160, NULL, 'chat', 1024),
    ('agent-001', 'success', 156, 498.0, 11.0, 120, NULL, 'chat', 768),
    ('agent-002', 'success', 678, 520.0, 25.0, 400, NULL, 'file_upload', 6144),
    ('agent-003', 'success', 234, 450.0, 14.0, 190, NULL, 'chat', 1280),
    ('agent-001', 'success', 312, 540.0, 18.0, 210, NULL, 'tool_call', 256),
    ('agent-002', 'error', 1500, 650.0, 55.0, 30, 'TIMEOUT', 'tool_call', 0);

-- Inserir dados de referência (baseline estatística)
INSERT INTO feature_store.reference_data 
    (feature_name, mean_value, std_value, min_value, max_value)
VALUES
    ('latency_ms', 380.0, 320.0, 123.0, 1500.0),
    ('memory_usage_mb', 510.0, 60.0, 410.0, 650.0),
    ('cpu_percent', 20.0, 12.0, 8.0, 55.0),
    ('grok_tokens_used', 220.0, 100.0, 30.0, 400.0);

-- View para monitoramento
CREATE OR REPLACE VIEW feature_store.vw_monitoring AS
SELECT
    DATE_TRUNC('hour', timestamp) AS hour,
    agent_id,
    COUNT(*) AS request_count,
    AVG(latency_ms) AS avg_latency,
    MAX(latency_ms) AS max_latency,
    SUM(CASE WHEN has_error = 1 THEN 1 ELSE 0 END) AS error_count,
    AVG(memory_usage_mb) AS avg_memory,
    AVG(cpu_percent) AS avg_cpu
FROM feature_store.vw_features_v1
GROUP BY DATE_TRUNC('hour', timestamp), agent_id
ORDER BY hour DESC;

-- Comentários documentais
COMMENT ON SCHEMA feature_store IS 'Feature store para ML pipeline';
COMMENT ON TABLE feature_store.raw_events IS 'Eventos brutos do agente WASM';
COMMENT ON VIEW feature_store.vw_features_v1 IS 'Features v1 - baseline';
COMMENT ON VIEW feature_store.vw_features_v2 IS 'Features v2 - com drift';
