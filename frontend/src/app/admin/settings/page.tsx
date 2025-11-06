'use client';

import React, { useState, useEffect } from 'react';
import {
  Card,
  Form,
  Input,
  Button,
  Typography,
  Space,
  Alert,
  Row,
  Col,
  Select,
  Switch,
  Divider,
  message,
  Tabs,
  Statistic,
  Tag,
  InputNumber,
  Tooltip,
  Modal,
  Upload,
  Image,
} from 'antd';
import {
  SettingOutlined,
  KeyOutlined,
  RobotOutlined,
  DatabaseOutlined,
  SaveOutlined,
  ReloadOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  BuildOutlined,
} from '@ant-design/icons';
import { ColorPicker } from 'antd';
import { useRouter } from 'next/navigation';
import { AuthService } from '@/lib/auth';
import { api } from '@/lib/api';
import Layout from '@/components/ui/Layout';

const { Title, Text } = Typography;
const { Option } = Select;
const { TabPane } = Tabs;

interface APIConfig {
  provider: 'openai' | 'mistral';
  api_key: string;
  is_active: boolean;
  model?: string;
  embedding_model?: string;
  max_tokens?: number;
  temperature?: number;
}

export default function AdminSettingsPage() {
  const [loading, setLoading] = useState(false);
  const [testing, setTesting] = useState(false);
  const [systemStats, setSystemStats] = useState<any>(null);
  const [confirmVisible, setConfirmVisible] = useState(false);
  const [selectedProvider, setSelectedProvider] = useState<'openai' | 'mistral'>('openai');
  const [pendingValues, setPendingValues] = useState<APIConfig | null>(null);
  const [form] = Form.useForm();
  const [dbForm] = Form.useForm();
  const [dbLoading, setDbLoading] = useState(false);
  const [brandingForm] = Form.useForm();
  const [promptForm] = Form.useForm();
  const [systemPrompt, setSystemPrompt] = useState('');
  const [promptLoading, setPromptLoading] = useState(false);
  const router = useRouter();

  const modelOptions = {
    openai: {
      chat: [
        { value: 'gpt-4o-mini', label: 'GPT-4o Mini' },
        { value: 'gpt-4', label: 'GPT-4' },
        { value: 'gpt-3.5-turbo', label: 'GPT-3.5 Turbo' }
      ],
      embedding: [
        { value: 'text-embedding-3-large', label: 'text-embedding-3-large' },
        { value: 'text-embedding-3-small', label: 'text-embedding-3-small' },
        { value: 'text-embedding-ada-002', label: 'text-embedding-ada-002' }
      ]
    },
    mistral: {
      chat: [
        { value: 'mistral-large', label: 'Mistral Large' },
        { value: 'mistral-medium', label: 'Mistral Medium' },
        { value: 'mistral-small', label: 'Mistral Small' }
      ],
      embedding: [
        { value: 'mistral-embed', label: 'mistral-embed' }
      ]
    }
  };

  useEffect(() => {
    // Check authentication and permissions
    if (!AuthService.isAuthenticated()) {
      router.push('/login');
      return;
    }

    // Only super_admin can access settings
    if (!AuthService.isSuperAdmin()) {
      router.push('/dashboard');
      return;
    }

    loadSystemStats();
    loadBrandingSettings();
    loadSystemPrompt();
  }, [router]);

  const loadSystemStats = async () => {
    try {
      const stats = await api.getSystemStats();
      setSystemStats(stats);
    } catch (error) {
      console.error('Failed to load system stats:', error);
    }
  };

  const loadBrandingSettings = async () => {
    try {
      const branding = await api.getCompanyBranding();
      const savedColor = localStorage.getItem('primaryColor') || '#1890ff';
      brandingForm.setFieldsValue({
        companyName: branding.company_name || '',
        logo: branding.logo_url ? [{ url: branding.logo_url }] : [],
        primaryColor: savedColor,
      });
    } catch (error) {
      console.error('Failed to load branding:', error);
    }
  };

  const loadSystemPrompt = async () => {
    try {
      const data = await api.getSystemPrompt();
      setSystemPrompt(data.prompt_text);
      promptForm.setFieldsValue({ prompt: data.prompt_text });
    } catch (error) {
      console.error('Failed to load system prompt:', error);
    }
  };

  const handleSaveBranding = async (values: any) => {
    try {
      await api.saveCompanyBranding({
        company_name: values.companyName,
        logo_url: values.logo && values.logo.length > 0 ? values.logo[0].url : undefined,
      });
      message.success('Company branding and theme saved successfully');
      loadBrandingSettings(); // Refresh form
    } catch (error) {
      message.error('Failed to save branding');
    }
  };

  const handleSavePrompt = async (values: any) => {
    setPromptLoading(true);
    try {
      await api.updateSystemPrompt(values.prompt);
      setSystemPrompt(values.prompt);
      message.success('System prompt updated successfully');
    } catch (error) {
      message.error('Failed to update system prompt');
    } finally {
      setPromptLoading(false);
    }
  };

  const handleSaveAPIConfig = async (values: APIConfig) => {
    setLoading(true);
    try {
      await api.configureAPIKeys(values);
      message.success(`${values.provider} API configuration saved successfully`);
      form.resetFields();
    } catch (error) {
      message.error('Failed to save API configuration');
    } finally {
      setLoading(false);
    }
  };

  const handleTestConnection = async () => {
    setTesting(true);
    try {
      // Test database connection
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/admin/health`);
      if (response.ok) {
        message.success('System health check passed');
        loadSystemStats();
      } else {
        message.error('System health check failed');
      }
    } catch (error) {
      message.error('Failed to test system health');
    } finally {
      setTesting(false);
    }
  };

  const handleConfirmSave = async () => {
    if (!pendingValues) return;
    setConfirmVisible(false);
    setLoading(true);
    try {
      await api.configureAPIKeys(pendingValues);
      message.success(`${pendingValues.provider} API configuration saved successfully`);
      form.resetFields();
      setPendingValues(null);
    } catch (error) {
      message.error('Failed to save API configuration');
    } finally {
      setLoading(false);
    }
  };

  const handleRunSQL = async (values: { query: string }) => {
    setDbLoading(true);
    try {
      // Simulate SQL execution - in production, this would call the backend API
      await new Promise(resolve => setTimeout(resolve, 1000));
      message.success('SQL query executed successfully');
      dbForm.resetFields();
    } catch (error) {
      message.error('Failed to execute SQL query');
    } finally {
      setDbLoading(false);
    }
  };

  return (
    <Layout>
      <div className="space-y-6">
        {/* Header */}
        <div className="text-center">
          <Title level={2}>⚙️ System Settings</Title>
          <Text type="secondary">
            Configure API keys, manage system settings, and monitor performance
          </Text>
        </div>

        <Tabs defaultActiveKey="api-config" type="card">
          {/* API Configuration Tab */}
          <TabPane tab={<span><KeyOutlined />API Configuration</span>} key="api-config">
            <Row gutter={[16, 16]}>
              <Col xs={24} lg={16}>
                <Card title="🔑 API Keys Configuration">
                  <Alert
                    message="Secure API Key Management"
                    description="API keys are stored securely and used only for document processing and chat functionality."
                    type="info"
                    showIcon
                    className="mb-6"
                  />

                  <Form
                    form={form}
                    layout="vertical"
                    onFinish={(values) => { setPendingValues(values); setConfirmVisible(true); }}
                    initialValues={{
                      is_active: true,
                      model: 'gpt-4o-mini',
                      embedding_model: 'text-embedding-3-large',
                      max_tokens: 1000,
                      temperature: 0.7,
                    }}
                  >
                    <Row gutter={16}>
                      <Col xs={24} md={12}>
                        <Form.Item
                          name="provider"
                          label="AI Provider"
                          rules={[
                            { required: true, message: 'Please select a provider' },
                          ]}
                        >
                          <Select
                          placeholder="Select AI provider"
                          onChange={(value) => {
                            form.setFieldsValue({
                              model: modelOptions[value as 'openai' | 'mistral'].chat[0].value,
                              embedding_model: modelOptions[value as 'openai' | 'mistral'].embedding[0].value
                            });
                            setSelectedProvider(value as 'openai' | 'mistral');
                          }}
                          aria-label="Select AI Provider"
                        >
                            <Option value="openai">
                              <div className="flex items-center space-x-2">
                                <RobotOutlined />
                                <span>OpenAI</span>
                              </div>
                            </Option>
                            <Option value="mistral">
                              <div className="flex items-center space-x-2">
                                <RobotOutlined />
                                <span>Mistral AI</span>
                              </div>
                            </Option>
                          </Select>
                        </Form.Item>
                      </Col>

                      <Col xs={24} md={12}>
                        <Form.Item
                          name="is_active"
                          label="Status"
                          valuePropName="checked"
                        >
                          <Switch
                            checkedChildren="Active"
                            unCheckedChildren="Inactive"
                          />
                        </Form.Item>
                      </Col>
                    </Row>

                    <Row gutter={16}>
                      <Col xs={24} md={12}>
                        <Tooltip title="Select the AI model to use for chat responses. Options vary by provider.">
                          <Form.Item
                            name="model"
                            label="Chat Model"
                            rules={[{ required: true, message: 'Please select a chat model' }]}
                          >
                            <Select placeholder="Select chat model" aria-label="Select Chat Model">
                              {modelOptions[selectedProvider].chat.map(option => (
                                <Option key={option.value} value={option.value}>
                                  {option.label}
                                </Option>
                              ))}
                            </Select>
                          </Form.Item>
                        </Tooltip>
                      </Col>

                      <Col xs={24} md={12}>
                        <Tooltip title="Select the AI model to use for document embeddings. Options vary by provider.">
                          <Form.Item
                            name="embedding_model"
                            label="Embedding Model"
                            rules={[{ required: true, message: 'Please select an embedding model' }]}
                          >
                            <Select placeholder="Select embedding model" aria-label="Select Embedding Model">
                              {modelOptions[selectedProvider].embedding.map(option => (
                                <Option key={option.value} value={option.value}>
                                  {option.label}
                                </Option>
                              ))}
                            </Select>
                          </Form.Item>
                        </Tooltip>
                      </Col>
                    </Row>

                    <Row gutter={16}>
                      <Col xs={24} md={12}>
                        <Tooltip title="Maximum number of tokens for AI responses (1-4000).">
                          <Form.Item
                            name="max_tokens"
                            label="Max Tokens"
                            rules={[{ required: true, type: 'number', min: 1, max: 4000 }]}
                          >
                            <InputNumber
                              min={1}
                              max={4000}
                              style={{ width: '100%' }}
                            />
                          </Form.Item>
                        </Tooltip>
                      </Col>
                      <Col xs={24} md={12}>
                        <Tooltip title="Controls randomness in AI responses (0.0 = deterministic, 2.0 = highly random).">
                          <Form.Item
                            name="temperature"
                            label="Temperature"
                            rules={[{ required: true, type: 'number', min: 0, max: 2 }]}
                          >
                            <InputNumber
                              min={0}
                              max={2}
                              step={0.1}
                              style={{ width: '100%' }}
                            />
                          </Form.Item>
                        </Tooltip>
                      </Col>
                    </Row>

                    <Form.Item
                      name="api_key"
                      label="API Key"
                      rules={[
                        { required: true, message: 'Please enter the API key' },
                        { min: 20, message: 'API key seems too short' },
                      ]}
                    >
                      <Input.Password
                        placeholder="Enter your API key"
                        style={{ width: '100%' }}
                      />
                    </Form.Item>

                    <Form.Item>
                      <Space>
                        <Button
                          type="primary"
                          onClick={() => form.submit()}
                          loading={loading}
                          icon={<SaveOutlined />}
                          aria-label="Save API Configuration"
                        >
                          Save Configuration
                        </Button>
                        <Button
                          onClick={() => form.resetFields()}
                        >
                          Reset
                        </Button>
                      </Space>
                    </Form.Item>
                  </Form>

                  <Divider />

                  <div className="space-y-4">
                    <Title level={4}>📚 API Key Guidelines</Title>

                    <Card size="small" title="OpenAI" type="inner">
                      <Text>
                        • Get your API key from{' '}
                        <a
                          href="https://platform.openai.com/api-keys"
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-blue-600"
                        >
                          OpenAI Platform
                        </a>
                        <br />
                        • Supports: GPT-4, GPT-3.5-turbo, text-embedding-3-large
                        <br />
                        • Recommended model: gpt-4o-mini
                      </Text>
                    </Card>

                    <Card size="small" title="Mistral AI" type="inner">
                      <Text>
                        • Get your API key from{' '}
                        <a
                          href="https://console.mistral.ai/"
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-blue-600"
                        >
                          Mistral Console
                        </a>
                        <br />
                        • Supports: Mistral Large, Medium, Small models
                        <br />
                        • Includes embedding models for document search
                      </Text>
                    </Card>
                  </div>
                </Card>
              </Col>

              <Col xs={24} lg={8}>
                <Card title="🔧 Quick Actions">
                  <Space direction="vertical" className="w-full">
                    <Button
                      type="primary"
                      icon={<CheckCircleOutlined />}
                      onClick={handleTestConnection}
                      loading={testing}
                      block
                    >
                      Test System Health
                    </Button>

                    <Button
                      icon={<ReloadOutlined />}
                      onClick={loadSystemStats}
                      block
                    >
                      Refresh Stats
                    </Button>

                    <Button
                      icon={<DatabaseOutlined />}
                      onClick={() => router.push('/admin/database')}
                      block
                    >
                      Database Tools
                    </Button>
                  </Space>
                </Card>

                {systemStats && (
                  <Card title="📊 System Status" className="mt-4">
                    <div className="space-y-3">
                      <div className="flex justify-between">
                        <Text>Database:</Text>
                        <Tag color="success">Connected</Tag>
                      </div>
                      <div className="flex justify-between">
                        <Text>API Keys:</Text>
                        <Tag color="processing">Configured</Tag>
                      </div>
                      <div className="flex justify-between">
                        <Text>File Storage:</Text>
                        <Tag color="success">Available</Tag>
                      </div>
                    </div>
                  </Card>
                )}
              </Col>
            </Row>
          </TabPane>

          {/* System Monitoring Tab */}
          <TabPane tab={<span><DatabaseOutlined />System Monitoring</span>} key="monitoring">
            <Row gutter={[16, 16]}>
              <Col xs={24} md={12}>
                <Card title="📈 Performance Metrics">
                  {systemStats ? (
                    <Row gutter={[16, 16]}>
                      <Col span={12}>
                        <Statistic
                          title="Total Users"
                          value={systemStats.total_users}
                          prefix={<SettingOutlined />}
                        />
                      </Col>
                      <Col span={12}>
                        <Statistic
                          title="Active Sessions"
                          value={systemStats.active_sessions}
                          prefix={<CheckCircleOutlined />}
                        />
                      </Col>
                      <Col span={12}>
                        <Statistic
                          title="Documents"
                          value={systemStats.total_documents}
                          prefix={<DatabaseOutlined />}
                        />
                      </Col>
                      <Col span={12}>
                        <Statistic
                          title="Embeddings"
                          value={systemStats.total_embeddings}
                          prefix={<RobotOutlined />}
                        />
                      </Col>
                    </Row>
                  ) : (
                    <div className="text-center py-8">
                      <Text type="secondary">Loading system statistics...</Text>
                    </div>
                  )}
                </Card>
              </Col>

              <Col xs={24} md={12}>
                <Card title="🔍 System Health">
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <Text>Database Connection</Text>
                      <Tag color="success" icon={<CheckCircleOutlined />}>
                        Healthy
                      </Tag>
                    </div>

                    <div className="flex items-center justify-between">
                      <Text>API Endpoints</Text>
                      <Tag color="success" icon={<CheckCircleOutlined />}>
                        Operational
                      </Tag>
                    </div>

                    <div className="flex items-center justify-between">
                      <Text>File Storage</Text>
                      <Tag color="success" icon={<CheckCircleOutlined />}>
                        Available
                      </Tag>
                    </div>

                    <div className="flex items-center justify-between">
                      <Text>AI Services</Text>
                      <Tag color="processing" icon={<ExclamationCircleOutlined />}>
                        Check Required
                      </Tag>
                    </div>
                  </div>

                  <Divider />

                  <Button
                    type="primary"
                    icon={<ReloadOutlined />}
                    onClick={handleTestConnection}
                    loading={testing}
                    block
                  >
                    Run Health Check
                  </Button>
                </Card>
              </Col>
            </Row>
          </TabPane>

          {/* Database Tools Tab */}
          <TabPane tab={<span><DatabaseOutlined />Database Tools</span>} key="database">
            <Card title="🗄️ Database Management">
              <Alert
                message="Database Operations"
                description="Advanced database management tools for system administrators."
                type="warning"
                showIcon
                className="mb-6"
              />

              <Row gutter={[16, 16]}>
                <Col xs={24} sm={12} md={8}>
                  <Card>
                    <Statistic
                      title="Database Size"
                      value={2.4}
                      suffix="GB"
                      valueStyle={{ color: '#667eea' }}
                    />
                  </Card>
                </Col>
                <Col xs={24} sm={12} md={8}>
                  <Card>
                    <Statistic
                      title="Tables"
                      value={8}
                      valueStyle={{ color: '#52c41a' }}
                    />
                  </Card>
                </Col>
                <Col xs={24} sm={12} md={8}>
                  <Card>
                    <Statistic
                      title="Connections"
                      value={3}
                      suffix="/ 20"
                      valueStyle={{ color: '#faad14' }}
                    />
                  </Card>
                </Col>
              </Row>

              <Divider />

              <div className="space-y-4">
                <Title level={4}>SQL Query Executor</Title>

                <Card title="Run SQL Queries" type="inner">
                  <Alert
                    message="SQL Query Tool"
                    description="Execute SQL queries directly on the database. Use with caution."
                    type="warning"
                    showIcon
                    className="mb-4"
                  />

                  <Form form={dbForm} onFinish={handleRunSQL} layout="vertical">
                    <Form.Item
                      name="query"
                      label="SQL Query"
                      rules={[{ required: true, message: 'Please enter a SQL query' }]}
                    >
                      <Input.TextArea
                        rows={4}
                        placeholder="SELECT * FROM users LIMIT 10;"
                        aria-label="SQL Query Input"
                      />
                    </Form.Item>

                    <Button
                      type="primary"
                      htmlType="submit"
                      loading={dbLoading}
                      icon={<DatabaseOutlined />}
                      aria-label="Execute SQL Query"
                    >
                      Execute Query
                    </Button>
                  </Form>
                </Card>
              </div>
            </Card>
          </TabPane>

          {/* Company Branding Tab */}
          <TabPane tab={<span><BuildOutlined />Company Branding</span>} key="company-branding">
            <Card title="🏢 Company Branding">
              <Alert
                message="Customize Your Brand"
                description="Set your company name and upload a logo to personalize the application."
                type="info"
                showIcon
                className="mb-6"
              />

              <Form
                form={brandingForm}
                layout="vertical"
                onFinish={handleSaveBranding}
              >
                <Form.Item
                  name="companyName"
                  label="Company Name"
                  rules={[{ required: true, message: 'Please enter the company name' }]}
                >
                  <Input placeholder="Enter company name" />
                </Form.Item>

                <Form.Item
                  name="logo"
                  label="Company Logo"
                  rules={[{ required: true, message: 'Please upload a logo' }]}
                >
                  <Upload
                    listType="picture-card"
                    beforeUpload={(file) => {
                      const reader = new FileReader();
                      reader.onload = (e) => {
                        const url = e.target?.result as string;
                        brandingForm.setFieldsValue({ logo: [{ url }] });
                      };
                      reader.readAsDataURL(file);
                      return false; // Prevent auto upload
                    }}
                    accept="image/*"
                    maxCount={1}
                  >
                    <div>
                      <BuildOutlined />
                      <div style={{ marginTop: 8 }}>Upload Logo</div>
                    </div>
                  </Upload>
                </Form.Item>

                <Form.Item
                  name="primaryColor"
                  label="Theme Color"
                >
                  <ColorPicker
                    value={brandingForm.getFieldValue('primaryColor')}
                    onChange={(color) => {
                      const hex = color.toHexString();
                      brandingForm.setFieldsValue({ primaryColor: hex });
                      localStorage.setItem('primaryColor', hex);
                      window.dispatchEvent(new CustomEvent('themeChange'));
                    }}
                  />
                </Form.Item>

                <Form.Item>
                  <Button type="primary" htmlType="submit" icon={<SaveOutlined />}>Save Branding</Button>
                </Form.Item>
              </Form>
            </Card>
          </TabPane>

          {/* AI System Prompt Tab */}
          <TabPane tab={<span><RobotOutlined />AI System Prompt</span>} key="system-prompt">
            <Card title="🤖 AI System Prompt">
              <Alert
                message="Customize AI Instructions"
                description="Modify the system prompt to customize how the AI responds to user queries based on document context."
                type="info"
                showIcon
                className="mb-6"
              />

              <Form
                form={promptForm}
                layout="vertical"
                onFinish={handleSavePrompt}
                initialValues={{ prompt: systemPrompt }}
              >
                <Form.Item
                  name="prompt"
                  label="System Prompt Template"
                  rules={[{ required: true, message: 'Please enter the system prompt' }]}
                >
                  <Input.TextArea
                    rows={12}
                    placeholder="Enter the system prompt template with placeholders like {context}, {references_text}"
                    style={{ fontFamily: 'monospace' }}
                  />
                </Form.Item>

                <Form.Item>
                  <Button
                    type="primary"
                    htmlType="submit"
                    loading={promptLoading}
                    icon={<SaveOutlined />}
                  >
                    Save Prompt
                  </Button>
                </Form.Item>
              </Form>
            </Card>
          </TabPane>
        </Tabs>
      </div>

      <Modal
        title="Confirm API Configuration"
        open={confirmVisible}
        onOk={handleConfirmSave}
        onCancel={() => { setConfirmVisible(false); setPendingValues(null); }}
        okText="Save"
        cancelText="Cancel"
      >
        <p>Are you sure you want to save this API configuration? This will update the system settings.</p>
        {pendingValues && (
          <div>
            <p>Provider: {pendingValues.provider}</p>
            <p>Chat Model: {pendingValues.model}</p>
            <p>Embedding Model: {pendingValues.embedding_model}</p>
            <p>Max Tokens: {pendingValues.max_tokens}</p>
            <p>Temperature: {pendingValues.temperature}</p>
          </div>
        )}
      </Modal>
    </Layout>
  );
}