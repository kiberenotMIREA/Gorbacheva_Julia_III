import { useState, useEffect } from 'react';
import {
  ConfigProvider, theme, Layout, Tabs, Card, Input, Select,
  Button, Progress, Tag, Alert, Typography, Space, Descriptions,
  Result, Flex,
} from 'antd';
import {
  SafetyCertificateOutlined, BugOutlined, ExperimentOutlined,
  CheckCircleOutlined, CloseCircleOutlined, LoadingOutlined,
  GithubOutlined,
} from '@ant-design/icons';

const { Header, Content, Footer } = Layout;
const { TextArea } = Input;
const { Title, Text, Paragraph } = Typography;

// =============================================================================
// Базовый URL API (через Nginx прокси /api/* -> FastAPI)
// =============================================================================
const API_BASE = '/api';

// =============================================================================
// Темная тема Ant Design
// =============================================================================
const darkTheme = {
  algorithm: theme.darkAlgorithm,
  token: {
    colorPrimary: '#1677ff',
    colorBgContainer: '#1e293b',
    colorBgElevated: '#1e293b',
    colorBorder: '#334155',
    colorText: '#e2e8f0',
    colorTextSecondary: '#94a3b8',
    borderRadius: 8,
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
  },
};

// =============================================================================
// Главный компонент приложения
// =============================================================================
export default function App() {
  // --- Состояния -----------------------------------------------------------
  const [health, setHealth] = useState(null);
  const [methods, setMethods] = useState([]);
  const [defaultMethod, setDefaultMethod] = useState('ensemble');
  const [code, setCode] = useState('');
  const [method, setMethod] = useState('ensemble');
  const [predicting, setPredicting] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // ===========================================================================
  // Эффект: загрузка /health и /methods при монтировании
  // ===========================================================================
  useEffect(() => {
    Promise.all([
      fetch(`${API_BASE}/health`).then(r => r.json()),
      fetch(`${API_BASE}/methods`).then(r => r.json()),
    ])
      .then(([h, m]) => {
        setHealth(h);
        setMethods(m.methods);
        setDefaultMethod(m.default);
        setMethod(m.default);
      })
      .catch(() => setHealth({ status: 'unreachable', model_loaded: false }));
  }, []);

  // ===========================================================================
  // Отправка запроса на /predict
  // ===========================================================================
  const handlePredict = async () => {
    const trimmed = code.trim();
    if (!trimmed || trimmed.length < 10) {
      setError('Код должен содержать минимум 10 символов.');
      return;
    }
    setPredicting(true);
    setResult(null);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code: trimmed, method }),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || `Ошибка ${res.status}`);
      }
      setResult(await res.json());
    } catch (e) {
      setError(e.message);
    } finally {
      setPredicting(false);
    }
  };

  // ===========================================================================
  // Статус сервера
  // ===========================================================================
  const healthStatus =
    !health ? { color: 'default', text: 'Загрузка...', icon: <LoadingOutlined /> }
      : health.status === 'healthy'
        ? { color: 'success', text: 'Сервис работает', icon: <CheckCircleOutlined /> }
        : { color: 'error', text: 'Сервис недоступен', icon: <CloseCircleOutlined /> };

  // ===========================================================================
  // Вкладки
  // ===========================================================================
  const tabItems = [
    {
      key: 'predict',
      label: (
        <span><ExperimentOutlined style={{ marginRight: 6 }} />Предсказание</span>
      ),
      children: (
        <Space direction="vertical" size="large" style={{ width: '100%' }}>
          {/* --- Ввод кода --- */}
          <Card
            title={
              <Title level={4} style={{ margin: 0, color: '#e2e8f0' }}>
                <BugOutlined style={{ marginRight: 8 }} />
                Введите код на C/C++ для анализа
              </Title>
            }
          >
            <TextArea
              rows={8}
              placeholder={'void func() {\n  char buf[10];\n  strcpy(buf, input);\n}'}
              value={code}
              onChange={e => setCode(e.target.value)}
              style={{ fontFamily: '"Fira Code", "Cascadia Code", "Consolas", monospace', fontSize: 13 }}
            />
            <Flex gap="middle" align="center" style={{ marginTop: 16 }} wrap="wrap">
              <Flex gap="small" align="center" wrap="wrap">
                <Text type="secondary">Метод:</Text>
                <Select
                  value={method}
                  onChange={setMethod}
                  style={{ minWidth: 220 }}
                  options={methods.map(m => ({ value: m.id, label: m.name }))}
                  popupMatchSelectWidth={false}
                />
              </Flex>
              <Button
                type="primary"
                size="large"
                icon={predicting ? <LoadingOutlined /> : <SafetyCertificateOutlined />}
                onClick={handlePredict}
                loading={predicting}
                disabled={!code.trim()}
              >
                {predicting ? 'Анализ...' : 'Анализировать'}
              </Button>
            </Flex>
          </Card>

          {/* --- Результат --- */}
          {error && (
            <Alert
              message="Ошибка"
              description={error}
              type="error"
              showIcon
              closable
              onClose={() => setError(null)}
            />
          )}

          {result && (
            <Card
              title={
                <Title level={4} style={{ margin: 0, color: '#e2e8f0' }}>
                  <SafetyCertificateOutlined style={{ marginRight: 8 }} />
                  Результат
                </Title>
              }
            >
              <Result
                status={result.label === 'vulnerable' ? 'error' : 'success'}
                title={
                  result.label === 'vulnerable'
                    ? '⚠ Обнаружена потенциальная уязвимость'
                    : '✅ Код безопасен'
                }
                subTitle={`Уверенность модели: ${(result.confidence * 100).toFixed(1)}%`}
                style={{ padding: '16px 0' }}
              />

              {/* Вероятности классов */}
              <Space direction="vertical" style={{ width: '100%', marginBottom: 16 }}>
                <Text strong style={{ color: '#94a3b8' }}>Вероятности классов</Text>
                <div>
                  <Flex justify="space-between" style={{ marginBottom: 4 }}>
                    <Text>✅ Безопасный</Text>
                    <Text strong>{(result.probabilities[0] * 100).toFixed(1)}%</Text>
                  </Flex>
                  <Progress
                    percent={+(result.probabilities[0] * 100).toFixed(1)}
                    strokeColor="#22c55e"
                    trailColor="#334155"
                    showInfo={false}
                    size="small"
                  />
                </div>
                <div>
                  <Flex justify="space-between" style={{ marginBottom: 4 }}>
                    <Text>⚠ Уязвимый</Text>
                    <Text strong>{(result.probabilities[1] * 100).toFixed(1)}%</Text>
                  </Flex>
                  <Progress
                    percent={+(result.probabilities[1] * 100).toFixed(1)}
                    strokeColor="#ef4444"
                    trailColor="#334155"
                    showInfo={false}
                    size="small"
                  />
                </div>
              </Space>

              {/* Детали */}
              <Descriptions
                column={{ xs: 1, sm: 2 }}
                size="small"
                bordered
                styles={{
                  label: { color: '#94a3b8', fontWeight: 500 },
                  content: { color: '#e2e8f0' },
                }}
              >
                <Descriptions.Item label="Время инференса">
                  {result.inference_time_ms} мс
                </Descriptions.Item>
                <Descriptions.Item label="Метод">
                  {methods.find(m => m.id === result.method)?.name || result.method}
                </Descriptions.Item>
              </Descriptions>
            </Card>
          )}
        </Space>
      ),
    },
    {
      key: 'methods',
      label: (
        <span><ExperimentOutlined style={{ marginRight: 6 }} />Методы</span>
      ),
      children: (
        <Space direction="vertical" size="middle" style={{ width: '100%' }}>
          {methods.length === 0 && (
            <Card><Text type="secondary">Загрузка методов...</Text></Card>
          )}
          {methods.map(m => (
            <Card key={m.id} size="small" hoverable>
              <Flex gap="small" align="center" wrap="wrap" style={{ marginBottom: 8 }}>
                <Tag color="blue" style={{ fontFamily: 'monospace', fontWeight: 700 }}>{m.id}</Tag>
                <Text strong style={{ color: '#e2e8f0', fontSize: 16 }}>{m.name}</Text>
                {m.id === defaultMethod && (
                  <Tag color="cyan">по умолчанию</Tag>
                )}
              </Flex>
              <Paragraph type="secondary" style={{ marginBottom: 12 }}>
                {m.description}
              </Paragraph>
              <Flex gap="4px" wrap="wrap">
                {m.base_models.map(model => (
                  <Tag key={model} color="default" style={{ color: '#cbd5e1' }}>
                    {model}
                  </Tag>
                ))}
              </Flex>
            </Card>
          ))}
        </Space>
      ),
    },
  ];

  // ===========================================================================
  // Render
  // ===========================================================================
  return (
    <ConfigProvider theme={darkTheme}>
      <Layout style={{ minHeight: '100vh', backgroundColor: '#0f172a' }}>
        {/* Header */}
        <Header
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            padding: '0 24px',
            height: 64,
            borderBottom: '1px solid #334155',
            backgroundColor: '#1e293b',
          }}
        >
          <Flex vertical gap={0}>
            <Title level={3} style={{ margin: 0, color: '#e2e8f0' }}>
              🛡️ Vulnerability Scoring
            </Title>
            <Text type="secondary" style={{ fontSize: 12, lineHeight: 1.2 }}>
              Обнаружение уязвимостей в коде C/C++
            </Text>
          </Flex>
          <Tag
            icon={healthStatus.icon}
            color={healthStatus.color}
            style={{ fontSize: 13, padding: '4px 12px' }}
          >
            {healthStatus.text}
          </Tag>
        </Header>

        {/* Content */}
        <Content style={{ maxWidth: 1100, margin: '0 auto', padding: '24px 24px 0', width: '100%' }}>
          <Tabs
            defaultActiveKey="predict"
            items={tabItems}
            size="large"
            style={{ color: '#e2e8f0' }}
          />
        </Content>

        {/* Footer */}
        <Footer
          style={{
            textAlign: 'center',
            padding: '12px 24px',
            fontSize: 12,
            color: '#475569',
            borderTop: '1px solid #334155',
            backgroundColor: '#0f172a',
          }}
        >
          Vulnerability Scoring
        </Footer>
      </Layout>
    </ConfigProvider>
  );
}
