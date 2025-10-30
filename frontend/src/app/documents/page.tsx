'use client';

import React, { useState, useEffect } from 'react';
import {
  Card,
  Table,
  Button,
  Upload,
  Modal,
  Typography,
  Space,
  Tag,
  Tooltip,
  message,
  Popconfirm,
  Input,
  Select,
  Row,
  Col,
  Statistic,
  Progress,
  notification,
} from 'antd';
import {
  UploadOutlined,
  DeleteOutlined,
  EyeOutlined,
  DownloadOutlined,
  SearchOutlined,
  ReloadOutlined,
  FileTextOutlined,
  InboxOutlined,
} from '@ant-design/icons';
import { useRouter } from 'next/navigation';
import { AuthService } from '@/lib/auth';
import { api } from '@/lib/api';
import Layout from '@/components/ui/Layout';
import type { Document, User } from '@/types';

const { Title, Text } = Typography;
const { Dragger } = Upload;
const { Search } = Input;
const { Option } = Select;

export default function DocumentsPage() {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [filteredDocuments, setFilteredDocuments] = useState<Document[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [user, setUser] = useState<User | null>(null);
  const [searchText, setSearchText] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [selectedDocument, setSelectedDocument] = useState<Document | null>(null);
  const [previewVisible, setPreviewVisible] = useState(false);
  const [selectedRowKeys, setSelectedRowKeys] = useState<React.Key[]>([]);
  const [processingStates, setProcessingStates] = useState({
    bulkExtract: false,
    bulkChunk: false,
    bulkEmbed: false,
    bulkDelete: false,
  });
  
  // Progress tracking state
  const [processingProgress, setProcessingProgress] = useState<{
    documentId: number | null;
    isProcessing: boolean;
    currentStage: 'extract' | 'chunk' | 'embed' | 'complete' | null;
    progress: number;
    chunksCreated?: number;
    embeddingsCreated?: number;
    processingTime?: number;
  }>({
    documentId: null,
    isProcessing: false,
    currentStage: null,
    progress: 0,
  });

  // Completion notification state
  const [completionModalVisible, setCompletionModalVisible] = useState(false);
  const [completionStats, setCompletionStats] = useState<{
    chunksCreated: number;
    embeddingsCreated: number;
    processingTime: number;
  } | null>(null);

  const router = useRouter();

  const columns = [
    {
      title: 'Document',
      dataIndex: 'original_filename',
      key: 'original_filename',
      render: (text: string, record: Document) => (
        <div className="flex items-center space-x-2">
          <FileTextOutlined />
          <div>
            <div className="font-medium">{text}</div>
            <div className="text-sm text-gray-500">{record.filename}</div>
          </div>
        </div>
      ),
    },
    {
      title: 'Size',
      dataIndex: 'file_size',
      key: 'file_size',
      render: (size: number) => `${(size / 1024).toFixed(1)} KB`,
      sorter: (a: Document, b: Document) => a.file_size - b.file_size,
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => {
        const statusConfig = {
          'not processed': { color: 'default', text: 'Not Processed' },
          'extracted': { color: 'processing', text: 'Extracted' },
          'chunked': { color: 'warning', text: 'Chunked' },
          'processed': { color: 'success', text: 'Processed' },
        };
        const config = statusConfig[status as keyof typeof statusConfig] || { color: 'default', text: status };
        return <Tag color={config.color}>{config.text}</Tag>;
      },
      filters: [
        { text: 'Not Processed', value: 'not processed' },
        { text: 'Extracted', value: 'extracted' },
        { text: 'Chunked', value: 'chunked' },
        { text: 'Processed', value: 'processed' },
      ],
      onFilter: (value: any, record: Document) => record.status === value,
    },
    {
      title: 'Upload Date',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (date: string) => new Date(date).toLocaleString(),
      sorter: (a: Document, b: Document) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime(),
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (record: Document) => (
        <Space>
          <Tooltip title="Preview">
            <Button
              size="small"
              icon={<EyeOutlined />}
              onClick={() => handlePreview(record)}
            />
          </Tooltip>

          {/* Single Process File button for unprocessed documents */}
          {record.status === 'not processed' && (
            <Tooltip title="Process document (Extract → Chunk → Embed)">
              <Button
                size="small"
                type="primary"
                onClick={() => handleProcessDocumentComplete(record.id)}
                loading={processingProgress.isProcessing && processingProgress.documentId === record.id}
                disabled={processingProgress.isProcessing && processingProgress.documentId !== record.id}
              >
                🚀 Process File
              </Button>
            </Tooltip>
          )}

          {/* Individual processing buttons for intermediate states */}
          {(record.status === 'not processed' || record.status === 'extracted') && (
            <Tooltip title="Chunk document">
              <Button
                size="small"
                type="default"
                onClick={() => handleDocumentAction(record.id, 'chunk')}
                disabled={record.status === 'not processed'}
              >
                Chunk
              </Button>
            </Tooltip>
          )}

          {(record.status === 'not processed' || record.status === 'extracted' || record.status === 'chunked') && (
            <Tooltip title="Embed document">
              <Button
                size="small"
                type="default"
                onClick={() => handleDocumentAction(record.id, 'embed')}
                disabled={record.status !== 'chunked'}
              >
                Embed
              </Button>
            </Tooltip>
          )}

          <Tooltip title="Chat about this document">
            <Button
              size="small"
              type="primary"
              onClick={() => handleChatWithDocument(record)}
              disabled={record.status !== 'processed'}
            >
              Chat
            </Button>
          </Tooltip>

          <Popconfirm
            title="Delete document"
            description="Are you sure you want to delete this document?"
            onConfirm={() => handleDeleteDocument(record.id)}
            okText="Yes"
            cancelText="No"
          >
            <Button
              size="small"
              danger
              icon={<DeleteOutlined />}
            />
          </Popconfirm>
        </Space>
      ),
    },
  ];

  useEffect(() => {
    // Check authentication
    if (!AuthService.isAuthenticated()) {
      router.push('/login');
      return;
    }

    // Load user and documents
    loadData();
  }, [router]);

  useEffect(() => {
    // Filter documents based on search and status
    let filtered = documents;

    if (searchText) {
      filtered = filtered.filter(doc =>
        doc.original_filename.toLowerCase().includes(searchText.toLowerCase()) ||
        doc.filename.toLowerCase().includes(searchText.toLowerCase())
      );
    }

    if (statusFilter !== 'all') {
      filtered = filtered.filter(doc => doc.status === statusFilter);
    }

    setFilteredDocuments(filtered);
  }, [documents, searchText, statusFilter]);

  const loadData = async () => {
    try {
      const [userData, documentsData] = await Promise.all([
        AuthService.getCurrentUser(),
        api.getDocuments(),
      ]);

      setUser(userData);
      setDocuments(documentsData);
    } catch (error) {
      message.error('Failed to load documents');
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (file: File) => {
    setUploading(true);

    // Check authentication before upload
    if (!AuthService.isAuthenticated()) {
      message.error('Please log in before uploading files');
      setUploading(false);
      return false;
    }

    console.log('Starting upload for file:', file.name);
    console.log('Is authenticated:', AuthService.isAuthenticated());
    console.log('Auth token present:', !!AuthService.getToken());
    console.log('Current user:', AuthService.getUser());

    try {
      await api.uploadDocument(file);
      message.success(`${file.name} uploaded successfully`);
      // Refresh documents list
      const updatedDocuments = await api.getDocuments();
      setDocuments(updatedDocuments);
    } catch (error) {
      console.error('Upload error:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      console.error('Upload error details:', {
        fileName: file.name,
        error: errorMessage,
        timestamp: new Date().toISOString()
      });
      message.error(`Failed to upload ${file.name}: ${errorMessage}`);
    } finally {
      setUploading(false);
    }
    return false; // Prevent default upload behavior
  };

  const handleDeleteDocument = async (documentId: number) => {
    try {
      await api.deleteDocument(documentId);
      message.success('Document deleted successfully');
      // Refresh documents list
      const updatedDocuments = await api.getDocuments();
      setDocuments(updatedDocuments);
    } catch (error) {
      message.error('Failed to delete document');
    }
  };

  const handlePreview = (document: Document) => {
    setSelectedDocument(document);
    setPreviewVisible(true);
  };

  const handleChatWithDocument = (document: Document) => {
    // Navigate to chat with the specific document selected
    router.push(`/chat?documentId=${document.id}`);
  };

  const handleDocumentAction = async (documentId: number, action: 'extract' | 'chunk' | 'embed') => {
    try {
      let result;
      switch (action) {
        case 'extract':
          result = await api.markDocumentExtracted(documentId);
          break;
        case 'chunk':
          result = await api.markDocumentChunked(documentId);
          break;
        case 'embed':
          result = await api.embedDocument(documentId);
          break;
      }

      message.success(`Document ${action}ed successfully`);
      // Refresh documents list
      const updatedDocuments = await api.getDocuments();
      setDocuments(updatedDocuments);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      console.error(`${action} error:`, {
        documentId,
        action,
        error: errorMessage,
        timestamp: new Date().toISOString()
      });
      message.error(`Failed to ${action} document: ${errorMessage}`);
    }
  };

  const handleProcessDocumentComplete = async (documentId: number) => {
    try {
      // Reset progress state
      setProcessingProgress({
        documentId,
        isProcessing: true,
        currentStage: 'extract',
        progress: 0,
      });

      // Start the complete processing pipeline (non-blocking)
      api.processDocumentComplete(documentId).then(result => {
        if (result.success) {
          // Final update when processing is complete
          setProcessingProgress({
            documentId,
            isProcessing: false,
            currentStage: 'complete',
            progress: 100,
            chunksCreated: result.metadata?.chunks_created,
            embeddingsCreated: result.metadata?.embeddings_created,
            processingTime: result.processing_time,
          });

          // Show completion notification
          setCompletionStats({
            chunksCreated: result.metadata?.chunks_created || 0,
            embeddingsCreated: result.metadata?.embeddings_created || 0,
            processingTime: result.processing_time || 0,
          });
          setCompletionModalVisible(true);

          message.success('Document processed successfully!');
          
          // Refresh documents list
          loadData();
        } else {
          setProcessingProgress({
            documentId: null,
            isProcessing: false,
            currentStage: null,
            progress: 0,
          });
          message.error('Document processing failed');
        }
      }).catch(error => {
        setProcessingProgress({
          documentId: null,
          isProcessing: false,
          currentStage: null,
          progress: 0,
        });
        const errorMessage = error instanceof Error ? error.message : 'Unknown error';
        console.error('Process document error:', {
          documentId,
          error: errorMessage,
          timestamp: new Date().toISOString()
        });
        message.error(`Failed to process document: ${errorMessage}`);
      });

      
    } catch (error) {
      setProcessingProgress({
        documentId: null,
        isProcessing: false,
        currentStage: null,
        progress: 0,
      });
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      console.error('Process document error:', {
        documentId,
        error: errorMessage,
        timestamp: new Date().toISOString()
      });
      message.error(`Failed to start document processing: ${errorMessage}`);
    }
  };

  const startProgressPolling = (documentId: number) => {
    let pollInterval: NodeJS.Timeout | null = null;
    
    pollInterval = setInterval(async () => {
      try {
        const status = await api.getProcessingStatus(documentId);
        
        // Update progress based on actual processing status
        let currentStage: 'extract' | 'chunk' | 'embed' | 'complete' | null = null;
        let progress = 0;

        switch (status.status) {
          case 'not processed':
            currentStage = 'extract';
            progress = 10;
            break;
          case 'extracted':
            currentStage = 'chunk';
            progress = 40;
            break;
          case 'chunked':
            currentStage = 'embed';
            progress = 70;
            break;
          case 'processed':
            currentStage = 'complete';
            progress = 100;
            break;
          case 'processing':
            // If processing, maintain current stage but show activity
            currentStage = processingProgress.currentStage || 'extract';
            progress = processingProgress.progress;
            break;
          case 'failed':
            currentStage = null;
            progress = 0;
            message.error('Document processing failed');
            if (pollInterval) clearInterval(pollInterval);
            break;
        }

        setProcessingProgress(prev => ({
          ...prev,
          currentStage,
          progress,
          chunksCreated: status.chunks_count,
          embeddingsCreated: status.embeddings_count,
        }));

        // Stop polling if processing is complete or failed
        if (status.status === 'processed' || status.status === 'failed') {
          if (pollInterval) clearInterval(pollInterval);
        }
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Unknown error';
        console.error('Error polling processing status:', {
          documentId,
          error: errorMessage,
          timestamp: new Date().toISOString()
        });
        // Continue polling even if there's an error
      }
    }, 2000); // Poll every 2 seconds

    // Return cleanup function
    return () => {
      if (pollInterval) clearInterval(pollInterval);
    };
  };



  // Cleanup polling when component unmounts or processing stops
  useEffect(() => {
    let cleanupPolling: (() => void) | null = null;
    
    if (processingProgress.isProcessing && processingProgress.documentId) {
      cleanupPolling = startProgressPolling(processingProgress.documentId);
    }

    return () => {
      if (cleanupPolling) cleanupPolling();
    };
  }, [processingProgress.isProcessing, processingProgress.documentId]);

  const handleBulkAction = async (action: 'extract' | 'chunk' | 'embed' | 'delete') => {
    if (selectedRowKeys.length === 0) {
      message.warning('Please select documents first');
      return;
    }

    const documentIds = selectedRowKeys.map(key => Number(key));
    setProcessingStates(prev => ({ ...prev, [`bulk${action.charAt(0).toUpperCase() + action.slice(1)}`]: true }));

    try {
      let result;
      switch (action) {
        case 'extract':
          result = await api.bulkExtractDocuments(documentIds);
          break;
        case 'chunk':
          result = await api.bulkChunkDocuments(documentIds);
          break;
        case 'embed':
          result = await api.bulkEmbedDocuments(documentIds);
          break;
        case 'delete':
          result = await api.bulkDeleteDocuments(documentIds);
          break;
      }

      message.success((result as any)?.message || `Bulk ${action} completed successfully`);
      setSelectedRowKeys([]); // Clear selection after successful operation
      // Refresh documents list
      const updatedDocuments = await api.getDocuments();
      setDocuments(updatedDocuments);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      console.error(`Bulk ${action} error:`, {
        documentIds,
        action,
        error: errorMessage,
        timestamp: new Date().toISOString()
      });
      message.error(`Failed to ${action} selected documents: ${errorMessage}`);
    } finally {
      setProcessingStates(prev => ({ ...prev, [`bulk${action.charAt(0).toUpperCase() + action.slice(1)}`]: false }));
    }
  };

  const handleBulkChat = () => {
    if (selectedRowKeys.length === 0) {
      message.warning('Please select documents first');
      return;
    }

    if (selectedRowKeys.length > 1) {
      message.warning('Please select only one document for chat');
      return;
    }

    const documentId = Number(selectedRowKeys[0]);
    const document = documents.find(doc => doc.id === documentId);

    if (document && document.status === 'processed') {
      router.push(`/chat?documentId=${documentId}`);
    } else {
      message.warning('Selected document must be fully embedded before chatting');
    }
  };

  const rowSelection = {
    selectedRowKeys,
    onChange: (newSelectedRowKeys: React.Key[]) => {
      setSelectedRowKeys(newSelectedRowKeys);
    },
    getCheckboxProps: (record: Document) => ({
      disabled: false, // You can add logic here to disable certain rows
    }),
  };

  const stats = {
    total: documents.length,
    notProcessed: documents.filter(doc => doc.status === 'not processed').length,
    extracted: documents.filter(doc => doc.status === 'extracted').length,
    chunked: documents.filter(doc => doc.status === 'chunked').length,
    processed: documents.filter(doc => doc.status === 'processed').length,
    totalSize: documents.reduce((acc, doc) => acc + doc.file_size, 0),
  };

  return (
    <Layout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex justify-between items-center">
          <div>
            <Title level={2}>📄 Document Management</Title>
            <Text type="secondary">
              Upload, manage, and chat with your documents
            </Text>
          </div>
          <Button
            type="primary"
            icon={<ReloadOutlined />}
            onClick={loadData}
            loading={loading}
          >
            Refresh
          </Button>
        </div>

        {/* Statistics */}
        <Row gutter={[16, 16]}>
          <Col xs={24} sm={12} md={6}>
            <Card>
              <Statistic
                title="Total Documents"
                value={stats.total}
                prefix={<FileTextOutlined />}
                valueStyle={{ color: '#667eea' }}
              />
            </Card>
          </Col>
          <Col xs={24} sm={12} md={6}>
            <Card>
              <Statistic
                title="Extracted"
                value={stats.extracted}
                valueStyle={{ color: '#1890ff' }}
              />
            </Card>
          </Col>
          <Col xs={24} sm={12} md={6}>
            <Card>
              <Statistic
                title="Chunked"
                value={stats.chunked}
                valueStyle={{ color: '#fa8c16' }}
              />
            </Card>
          </Col>
          <Col xs={24} sm={12} md={6}>
            <Card>
              <Statistic
                title="Total Size"
                value={stats.totalSize / (1024 * 1024)}
                precision={1}
                suffix="MB"
                valueStyle={{ color: '#722ed1' }}
              />
            </Card>
          </Col>
        </Row>

        {/* Bulk Actions */}
        {selectedRowKeys.length > 0 && (
          <Card title={`📋 Bulk Actions (${selectedRowKeys.length} selected)`}>
            <Row gutter={[16, 16]}>
              <Col xs={24} sm={8}>
                <Button
                  type="primary"
                  size="large"
                  block
                  onClick={() => handleBulkAction('extract')}
                  loading={processingStates.bulkExtract}
                  icon={<FileTextOutlined />}
                >
                  🚀 Bulk Extract
                </Button>
              </Col>
              <Col xs={24} sm={8}>
                <Button
                  size="large"
                  block
                  onClick={() => handleBulkAction('chunk')}
                  loading={processingStates.bulkChunk}
                  icon={<FileTextOutlined />}
                >
                  📦 Bulk Chunk
                </Button>
              </Col>
              <Col xs={24} sm={8}>
                <Button
                  size="large"
                  block
                  onClick={() => handleBulkAction('embed')}
                  loading={processingStates.bulkEmbed}
                  icon={<FileTextOutlined />}
                >
                  🧠 Bulk Embed
                </Button>
              </Col>
              <Col xs={24} sm={8}>
                <Button
                  type="primary"
                  size="large"
                  block
                  onClick={handleBulkChat}
                  icon={<FileTextOutlined />}
                >
                  💬 Chat with Selected
                </Button>
              </Col>
              <Col xs={24} sm={8}>
                <Popconfirm
                  title="Delete selected documents"
                  description={`Are you sure you want to delete ${selectedRowKeys.length} documents?`}
                  onConfirm={() => handleBulkAction('delete')}
                  okText="Yes"
                  cancelText="No"
                >
                  <Button
                    danger
                    size="large"
                    block
                    loading={processingStates.bulkDelete}
                    icon={<DeleteOutlined />}
                  >
                    🗑️ Bulk Delete
                  </Button>
                </Popconfirm>
              </Col>
              <Col xs={24} sm={8}>
                <Button
                  size="large"
                  block
                  onClick={() => setSelectedRowKeys([])}
                  icon={<FileTextOutlined />}
                >
                  Clear Selection
                </Button>
              </Col>
            </Row>
          </Card>
        )}

        {/* Progress Bar for Document Processing */}
        {processingProgress.isProcessing && (
          <Card title="🔄 Processing Document" className="border-blue-200">
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <Text strong>
                  {processingProgress.currentStage === 'extract' && '📄 Extracting document...'}
                  {processingProgress.currentStage === 'chunk' && '📦 Chunking document...'}
                  {processingProgress.currentStage === 'embed' && '🧠 Creating embeddings...'}
                  {processingProgress.currentStage === 'complete' && '✅ Processing complete!'}
                </Text>
                <Text type="secondary">{processingProgress.progress}%</Text>
              </div>
              
              <Progress
                percent={processingProgress.progress}
                status={processingProgress.currentStage === 'complete' ? 'success' : 'active'}
                strokeColor={{
                  '0%': '#108ee9',
                  '100%': '#87d068',
                }}
              />
              
              <div className="grid grid-cols-3 gap-4 text-center">
                <div className={`p-2 rounded ${processingProgress.currentStage === 'extract' ? 'bg-blue-100 border border-blue-300' : 'bg-gray-50'}`}>
                  <Text strong className={processingProgress.currentStage === 'extract' ? 'text-blue-600' : 'text-gray-500'}>
                    Extraction
                  </Text>
                </div>
                <div className={`p-2 rounded ${processingProgress.currentStage === 'chunk' ? 'bg-orange-100 border border-orange-300' : 'bg-gray-50'}`}>
                  <Text strong className={processingProgress.currentStage === 'chunk' ? 'text-orange-600' : 'text-gray-500'}>
                    Chunking
                  </Text>
                </div>
                <div className={`p-2 rounded ${processingProgress.currentStage === 'embed' ? 'bg-green-100 border border-green-300' : 'bg-gray-50'}`}>
                  <Text strong className={processingProgress.currentStage === 'embed' ? 'text-green-600' : 'text-gray-500'}>
                    Embedding
                  </Text>
                </div>
              </div>
            </div>
          </Card>
        )}

        {/* Upload Section */}
        <Card title="📁 Upload New Document">
          <Dragger
            name="file"
            multiple={true}
            beforeUpload={handleFileUpload}
            disabled={uploading}
            showUploadList={false}
          >
            <p className="ant-upload-drag-icon">
              <InboxOutlined />
            </p>
            <p className="ant-upload-text">
              Click or drag files to upload
            </p>
            <p className="ant-upload-hint">
              Support for PDF, DOCX, MD, HTML, and image files
            </p>
          </Dragger>
        </Card>

        {/* Filters and Search */}
        <Card>
          <Row gutter={[16, 16]} align="middle">
            <Col xs={24} sm={12} md={8}>
              <Search
                placeholder="Search documents..."
                value={searchText}
                onChange={(e) => setSearchText(e.target.value)}
                prefix={<SearchOutlined />}
              />
            </Col>
            <Col xs={24} sm={12} md={8}>
              <Select
                style={{ width: '100%' }}
                placeholder="Filter by status"
                value={statusFilter}
                onChange={setStatusFilter}
              >
                <Option value="all">All Status</Option>
                <Option value="not processed">Not Processed</Option>
                <Option value="extracted">Extracted</Option>
                <Option value="chunked">Chunked</Option>
                <Option value="processed">Processed</Option>
              </Select>
            </Col>
            <Col xs={24} sm={24} md={8}>
              <Text type="secondary">
                Showing {filteredDocuments.length} of {documents.length} documents
              </Text>
            </Col>
          </Row>
        </Card>

        {/* Documents Table */}
        <Card>
          <Table
            columns={columns}
            dataSource={filteredDocuments}
            loading={loading}
            rowKey="id"
            rowSelection={rowSelection}
            pagination={{
              showSizeChanger: true,
              showQuickJumper: true,
              showTotal: (total, range) =>
                `${range[0]}-${range[1]} of ${total} documents`,
            }}
          />
        </Card>

        {/* Document Preview Modal */}
        <Modal
          title={`Document: ${selectedDocument?.original_filename}`}
          open={previewVisible}
          onCancel={() => setPreviewVisible(false)}
          footer={[
            <Button key="close" onClick={() => setPreviewVisible(false)}>
              Close
            </Button>,
            <Button
              key="chat"
              type="primary"
              onClick={() => {
                setPreviewVisible(false);
                if (selectedDocument) handleChatWithDocument(selectedDocument);
              }}
            >
              Chat with Document
            </Button>,
          ]}
          width={800}
        >
          {selectedDocument && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Text strong>File Size:</Text>
                  <br />
                  <Text>{(selectedDocument.file_size / 1024).toFixed(1)} KB</Text>
                </div>
                <div>
                  <Text strong>Status:</Text>
                  <br />
                  <Tag color={
                    selectedDocument.status === 'processed' ? 'success' :
                    selectedDocument.status === 'extracted' ? 'processing' :
                    selectedDocument.status === 'chunked' ? 'warning' : 'default'
                  }>
                    {selectedDocument.status}
                  </Tag>
                </div>
                <div>
                  <Text strong>Upload Date:</Text>
                  <br />
                  <Text>{new Date(selectedDocument.created_at).toLocaleString()}</Text>
                </div>
                <div>
                  <Text strong>Type:</Text>
                  <br />
                  <Text>{selectedDocument.mime_type}</Text>
                </div>
              </div>

              {selectedDocument.status === 'processed' && (
                <div className="mt-4 p-4 bg-green-50 rounded-lg">
                  <Text strong className="text-green-800">
                    ✅ Document fully processed and ready for questions!
                  </Text>
                </div>
              )}

              {selectedDocument.status === 'extracted' && (
                <div className="mt-4 p-4 bg-blue-50 rounded-lg">
                  <Text strong className="text-blue-800">
                    📄 Document extracted successfully. Ready for chunking.
                  </Text>
                </div>
              )}

              {selectedDocument.status === 'chunked' && (
                <div className="mt-4 p-4 bg-orange-50 rounded-lg">
                  <Text strong className="text-orange-800">
                    📦 Document chunked successfully. Ready for embedding.
                  </Text>
                </div>
              )}

              {selectedDocument.status === 'not processed' && (
                <div className="mt-4 p-4 bg-gray-50 rounded-lg">
                  <Text strong className="text-gray-800">
                    📋 Document uploaded. Ready for extraction.
                  </Text>
                </div>
              )}
            </div>
          )}
        </Modal>

        {/* Completion Notification Modal */}
        <Modal
          title="🎉 Document Processing Complete!"
          open={completionModalVisible}
          onCancel={() => setCompletionModalVisible(false)}
          footer={[
            <Button key="close" onClick={() => setCompletionModalVisible(false)}>
              Close
            </Button>,
            <Button
              key="chat"
              type="primary"
              onClick={() => {
                setCompletionModalVisible(false);
                // Navigate to chat with the processed document
                if (processingProgress.documentId) {
                  const document = documents.find(doc => doc.id === processingProgress.documentId);
                  if (document) handleChatWithDocument(document);
                }
              }}
            >
              Chat with Document
            </Button>,
          ]}
          width={500}
        >
          {completionStats && (
            <div className="space-y-4">
              <div className="p-4 bg-green-50 rounded-lg">
                <Text strong className="text-green-800 text-lg">
                  ✅ Successfully processed your document!
                </Text>
              </div>
              
              <div className="grid grid-cols-2 gap-4">
                <div className="text-center p-3 bg-blue-50 rounded-lg">
                  <Text strong className="text-blue-700 text-xl">
                    {completionStats.chunksCreated}
                  </Text>
                  <br />
                  <Text type="secondary">Chunks Created</Text>
                </div>
                
                <div className="text-center p-3 bg-purple-50 rounded-lg">
                  <Text strong className="text-purple-700 text-xl">
                    {completionStats.embeddingsCreated}
                  </Text>
                  <br />
                  <Text type="secondary">Embeddings Generated</Text>
                </div>
              </div>
              
              <div className="text-center p-3 bg-orange-50 rounded-lg">
                <Text strong className="text-orange-700">
                  ⏱️ Total Processing Time: {(completionStats.processingTime / 60).toFixed(1)} minutes
                </Text>
              </div>
              
              <div className="mt-4 p-3 bg-gray-50 rounded-lg">
                <Text type="secondary">
                  Your document is now ready for intelligent conversations.
                  Click "Chat with Document" to start asking questions!
                </Text>
              </div>
            </div>
          )}
        </Modal>
      </div>
    </Layout>
  );
}