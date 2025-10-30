'use client';

import React, { useState, useEffect } from 'react';
import {
  Card,
  Typography,
  Row,
  Col,
  Button,
  Input,
  List,
  Tag,
  Space,
  message,
  Spin,
  Modal,
  Form,
  Select,
  Divider,
  Progress,
  Table,
  Tooltip,
  Popconfirm,
} from 'antd';
import {
  FileExcelOutlined,
  PlusOutlined,
  DeleteOutlined,
  DownloadOutlined,
  ReloadOutlined,
  QuestionCircleOutlined,
  UploadOutlined,
} from '@ant-design/icons';
import { useRouter } from 'next/navigation';
import { AuthService } from '@/lib/auth';
import { api } from '@/lib/api';
import Layout from '@/components/ui/Layout';

const { Title, Text, Paragraph } = Typography;
const { TextArea } = Input;
const { Option } = Select;

interface QuestionExport {
  id: number;
  filename: string;
  file_path: string;
  file_size: number;
  questions_count: number;
  document_ids: number[];
  created_at: string;
  status: string;
}

interface Document {
  id: number;
  original_filename: string;
  status: string;
}

interface ProgressData {
  session_id: number;
  user_id: number;
  total_questions: number;
  processed_questions: number;
  current_question: string;
  current_question_index: number;
  status: string;
  progress_percentage: number;
}

export default function QuestionExportPage() {
  const [questions, setQuestions] = useState<string[]>(['']);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [selectedDocuments, setSelectedDocuments] = useState<number[]>([]);
  const [exportName, setExportName] = useState('');
  const [loading, setLoading] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [exports, setExports] = useState<QuestionExport[]>([]);
  const [exportsLoading, setExportsLoading] = useState(false);
  const [progressData, setProgressData] = useState<ProgressData | null>(null);
  const [progressInterval, setProgressInterval] = useState<NodeJS.Timeout | null>(null);
  const [form] = Form.useForm();
  const router = useRouter();

  useEffect(() => {
    // Check authentication
    if (!AuthService.isAuthenticated()) {
      router.push('/login');
      return;
    }

    loadDocuments();
    loadExports();
  }, [router]);

  const loadDocuments = async () => {
    try {
      const docs = await api.getDocuments();
      setDocuments(docs);
    } catch (error) {
      console.error('Failed to load documents:', error);
      message.error('Failed to load documents');
    }
  };

  const loadExports = async () => {
    setExportsLoading(true);
    try {
      const userExports = await api.getUserExports();
      setExports(userExports);
    } catch (error) {
      console.error('Failed to load exports:', error);
      message.error('Failed to load exports');
    } finally {
      setExportsLoading(false);
    }
  };

  const addQuestion = () => {
    setQuestions([...questions, '']);
  };

  const updateQuestion = (index: number, value: string) => {
    const newQuestions = [...questions];
    newQuestions[index] = value;
    setQuestions(newQuestions);
  };

  const removeQuestion = (index: number) => {
    if (questions.length > 1) {
      const newQuestions = [...questions];
      newQuestions.splice(index, 1);
      setQuestions(newQuestions);
    }
  };

  const startProgressPolling = (sessionId: number) => {
    console.log('🚀 Starting progress polling for session:', sessionId);
    const interval = setInterval(async () => {
      try {
        const progress = await api.getProcessingProgress(sessionId);
        console.log('📊 Progress data received:', progress);
        setProgressData(progress);
        
        // If processing is complete, stop polling and refresh exports
        if (progress.status === 'completed' || progress.status === 'failed') {
          console.log('✅ Processing completed or failed, stopping polling');
          clearInterval(interval);
          setProgressInterval(null);
          setProgressData(null);
          loadExports();
        }
      } catch (error) {
        console.error('❌ Failed to fetch progress:', error);
        // If progress endpoint fails, stop polling
        clearInterval(interval);
        setProgressInterval(null);
        setProgressData(null);
      }
    }, 1000); // Poll every second
    
    setProgressInterval(interval);
  };

  const stopProgressPolling = () => {
    if (progressInterval) {
      clearInterval(progressInterval);
      setProgressInterval(null);
    }
    setProgressData(null);
  };

  const handleProcessQuestions = async () => {
    const validQuestions = questions.filter(q => q.trim().length > 0);
    
    if (validQuestions.length === 0) {
      message.error('Please add at least one question');
      return;
    }

    if (validQuestions.length > 100) {
      message.error('Maximum 100 questions allowed per export');
      return;
    }

    setProcessing(true);
    try {
      console.log('📤 Sending questions for processing:', validQuestions);
      const response = await api.processQuestionsAndExport({
        questions: validQuestions,
        document_ids: selectedDocuments.length > 0 ? selectedDocuments : undefined,
        export_name: exportName || undefined,
      });

      console.log('📥 Response received:', response);
      message.success('Question processing started! Tracking progress...');
      
      // Start progress polling if session_id is available
      if (response.session_id) {
        console.log('🎯 Session ID received:', response.session_id);
        startProgressPolling(response.session_id);
      } else {
        console.warn('⚠️ No session ID received from backend');
      }
      
      // Reset form
      setQuestions(['']);
      setSelectedDocuments([]);
      setExportName('');
      form.resetFields();

    } catch (error) {
      console.error('❌ Failed to process questions:', error);
      message.error('Failed to start question processing');
      stopProgressPolling();
    } finally {
      setProcessing(false);
    }
  };

  const handleDownload = async (exportId: number, filename: string) => {
    try {
      const blob = await api.downloadExportFile(exportId);
      
      // Create download link
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.style.display = 'none';
      a.href = url;
      a.download = filename;
      
      document.body.appendChild(a);
      a.click();
      
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      
      message.success('File downloaded successfully');
    } catch (error) {
      console.error('Download failed:', error);
      message.error('Failed to download file');
    }
  };

  const handleDeleteExport = async (exportId: number) => {
    try {
      await api.deleteExport(exportId);
      message.success('Export deleted successfully');
      loadExports();
    } catch (error) {
      console.error('Delete failed:', error);
      message.error('Failed to delete export');
    }
  };

  // Cleanup progress polling on component unmount
  useEffect(() => {
    return () => {
      if (progressInterval) {
        clearInterval(progressInterval);
      }
    };
  }, [progressInterval]);

  const columns = [
    {
      title: 'Filename',
      dataIndex: 'filename',
      key: 'filename',
      render: (filename: string) => (
        <Text strong>{filename}</Text>
      ),
    },
    {
      title: 'Questions',
      dataIndex: 'questions_count',
      key: 'questions_count',
      render: (count: number) => (
        <Tag color="blue">{count} questions</Tag>
      ),
    },
    {
      title: 'Size',
      dataIndex: 'file_size',
      key: 'file_size',
      render: (size: number) => (
        <Text type="secondary">{(size / 1024).toFixed(1)} KB</Text>
      ),
    },
    {
      title: 'Created',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (date: string) => (
        <Text type="secondary">{new Date(date).toLocaleDateString()}</Text>
      ),
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (record: QuestionExport) => (
        <Space>
          <Tooltip title="Download Excel file">
            <Button
              type="primary"
              icon={<DownloadOutlined />}
              size="small"
              onClick={() => handleDownload(record.id, record.filename)}
            >
              Download
            </Button>
          </Tooltip>
          <Popconfirm
            title="Delete this export?"
            description="Are you sure you want to delete this export? This action cannot be undone."
            onConfirm={() => handleDeleteExport(record.id)}
            okText="Yes"
            cancelText="No"
          >
            <Button danger icon={<DeleteOutlined />} size="small">
              Delete
            </Button>
          </Popconfirm>
        </Space>
      ),
    },
  ];

  return (
    <Layout>
      <div className="space-y-6">
        {/* Header */}
        <div className="text-center">
          <Title level={2}>
            <FileExcelOutlined className="mr-2" />
            Question Export Tool
          </Title>
          <Text type="secondary">
            Process multiple questions and generate Excel files with AI-generated answers
          </Text>
        </div>

        <Row gutter={[24, 24]}>
          {/* Question Input Section */}
          <Col xs={24} lg={12}>
            <Card
              title="📝 Enter Questions"
              extra={
                <Button
                  type="dashed"
                  icon={<PlusOutlined />}
                  onClick={addQuestion}
                >
                  Add Question
                </Button>
              }
            >
              <Form form={form} layout="vertical">
                <Form.Item label="Export Name (Optional)">
                  <Input
                    placeholder="e.g., Customer Support Questions"
                    value={exportName}
                    onChange={(e) => setExportName(e.target.value)}
                  />
                </Form.Item>

                <Form.Item label="Select Documents (Optional)">
                  <Select
                    mode="multiple"
                    placeholder="Select documents to use as context"
                    value={selectedDocuments}
                    onChange={setSelectedDocuments}
                    allowClear
                  >
                    {documents
                      .filter(doc => doc.status === 'processed')
                      .map(doc => (
                        <Option key={doc.id} value={doc.id}>
                          {doc.original_filename}
                        </Option>
                      ))}
                  </Select>
                  <Text type="secondary" className="text-xs">
                    Selected documents will be used as context for generating answers
                  </Text>
                </Form.Item>

                <Divider />

                <Form.Item label="Questions">
                  <div className="space-y-3">
                    {questions.map((question, index) => (
                      <div key={index} className="flex gap-2">
                        <TextArea
                          placeholder={`Question ${index + 1}`}
                          value={question}
                          onChange={(e) => updateQuestion(index, e.target.value)}
                          rows={2}
                          style={{ flex: 1 }}
                        />
                        {questions.length > 1 && (
                          <Button
                            danger
                            type="text"
                            icon={<DeleteOutlined />}
                            onClick={() => removeQuestion(index)}
                          />
                        )}
                      </div>
                    ))}
                  </div>
                </Form.Item>

                <Form.Item>
                  <Button
                    type="primary"
                    icon={<FileExcelOutlined />}
                    loading={processing}
                    onClick={handleProcessQuestions}
                    block
                    size="large"
                    disabled={!!progressData}
                  >
                    {processing ? 'Processing Questions...' : 'Generate Excel Export'}
                  </Button>
                  
                  {/* Progress Display */}
                  {progressData && (
                    <div className="mt-4 p-4 border rounded-lg bg-blue-50">
                      <div className="flex justify-between items-center mb-2">
                        <Text strong>Processing Progress</Text>
                        <Text type="secondary">
                          {progressData.processed_questions}/{progressData.total_questions} questions
                        </Text>
                      </div>
                      
                      <Progress
                        percent={Math.round(progressData.progress_percentage)}
                        status={progressData.status === 'failed' ? 'exception' : 'active'}
                        strokeColor={{
                          '0%': '#108ee9',
                          '100%': '#87d068',
                        }}
                      />
                      
                      <div className="mt-2 text-sm">
                        <Text type="secondary">
                          Current: {progressData.current_question}
                        </Text>
                      </div>
                      
                      {progressData.status === 'completed' && (
                        <div className="mt-2">
                          <Text type="success">✅ Processing completed!</Text>
                        </div>
                      )}
                      
                      {progressData.status === 'failed' && (
                        <div className="mt-2">
                          <Text type="danger">❌ Processing failed</Text>
                        </div>
                      )}
                    </div>
                  )}
                </Form.Item>
              </Form>
            </Card>
          </Col>

          {/* Instructions and Info */}
          <Col xs={24} lg={12}>
            <Card title="ℹ️ How It Works">
              <div className="space-y-4">
                <div>
                  <Text strong>Step 1: Enter Questions</Text>
                  <Paragraph type="secondary" className="mt-1">
                    Add one or more questions you want answered. Each question will be processed by the AI system.
                  </Paragraph>
                </div>

                <div>
                  <Text strong>Step 2: Select Documents (Optional)</Text>
                  <Paragraph type="secondary" className="mt-1">
                    Choose processed documents to provide context for more accurate answers.
                  </Paragraph>
                </div>

                <div>
                  <Text strong>Step 3: Generate Export</Text>
                  <Paragraph type="secondary" className="mt-1">
                    The system will process all questions, generate AI answers, and create a structured Excel file.
                  </Paragraph>
                </div>

                <div>
                  <Text strong>Step 4: Download Results</Text>
                  <Paragraph type="secondary" className="mt-1">
                    Download the Excel file containing all questions and their AI-generated answers.
                  </Paragraph>
                </div>

                <Divider />

                <div className="bg-blue-50 p-4 rounded-lg">
                  <Text strong>💡 Tips:</Text>
                  <ul className="list-disc list-inside mt-2 space-y-1 text-sm">
                    <li>Keep questions clear and specific for better answers</li>
                    <li>Use document context for domain-specific questions</li>
                    <li>Maximum 100 questions per export</li>
                    <li>Processing time depends on the number of questions</li>
                  </ul>
                </div>
              </div>
            </Card>
          </Col>
        </Row>

        {/* Previous Exports */}
        <Card
          title="📁 Previous Exports"
          extra={
            <Button
              icon={<ReloadOutlined />}
              onClick={loadExports}
              loading={exportsLoading}
            >
              Refresh
            </Button>
          }
        >
          {exportsLoading ? (
            <div className="text-center py-8">
              <Spin size="large" />
              <div className="mt-4">
                <Text>Loading exports...</Text>
              </div>
            </div>
          ) : exports.length > 0 ? (
            <Table
              dataSource={exports}
              columns={columns}
              rowKey="id"
              pagination={{ pageSize: 10 }}
              size="middle"
            />
          ) : (
            <div className="text-center py-8">
              <FileExcelOutlined style={{ fontSize: '48px', color: '#d9d9d9' }} />
              <div className="mt-4">
                <Title level={4} type="secondary">No exports yet</Title>
                <Text type="secondary">
                  Create your first question export using the form above.
                </Text>
              </div>
            </div>
          )}
        </Card>
      </div>
    </Layout>
  );
}