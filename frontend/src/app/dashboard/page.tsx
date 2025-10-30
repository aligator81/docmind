'use client';

import React, { useEffect, useState } from 'react';
import { Card, Typography, Row, Col, Statistic, Button, Upload, message, Spin, Tag } from 'antd';
import {
  FileTextOutlined,
  MessageOutlined,
  UserOutlined,
  UploadOutlined,
  InboxOutlined,
  PlayCircleOutlined,
} from '@ant-design/icons';
import { useRouter } from 'next/navigation';
import { AuthService } from '@/lib/auth';
import { api } from '@/lib/api';
import Layout from '@/components/ui/Layout';
import type { Document, User } from '@/types';

const { Title, Text } = Typography;
const { Dragger } = Upload;

export default function DashboardPage() {
  const [user, setUser] = useState<User | null>(null);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [processingDocuments, setProcessingDocuments] = useState<Set<number>>(new Set());
  const router = useRouter();

  useEffect(() => {
    // Check authentication
    if (!AuthService.isAuthenticated()) {
      router.push('/login');
      return;
    }

    // Load user data and documents
    loadDashboardData();
  }, [router]);

  // Debug logging - only in client side
  useEffect(() => {
    if (typeof window !== 'undefined') {
      console.log('Dashboard loaded, user authenticated:', AuthService.isAuthenticated());
      console.log('User data:', AuthService.getUser());
    }
  }, []);

  const loadDashboardData = async () => {
    try {
      // First try to get user data from localStorage (faster)
      const localUser = AuthService.getUser();
      if (localUser) {
        setUser(localUser);
      }

      // Then try to get fresh data from API
      const [userData, documentsData] = await Promise.all([
        AuthService.getCurrentUser().catch(() => localUser), // Fallback to localStorage if API fails
        api.getDocuments(),
      ]);

      if (userData) {
        setUser(userData);
      }
      setDocuments(documentsData);
    } catch (error) {
      console.error('Dashboard load error:', error);
      message.error('Failed to load dashboard data');
      // Still try to use localStorage data if available
      const localUser = AuthService.getUser();
      if (localUser) {
        setUser(localUser);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (file: File) => {
    setUploading(true);
    try {
      await api.uploadDocument(file);
      message.success(`${file.name} uploaded successfully`);
      // Refresh documents list
      const updatedDocuments = await api.getDocuments();
      setDocuments(updatedDocuments);
    } catch (error) {
      message.error(`Failed to upload ${file.name}`);
    } finally {
      setUploading(false);
    }
    return false; // Prevent default upload behavior
  };

  if (loading) {
    return (
      <Layout>
        <div className="flex justify-center items-center h-64">
          <div className="text-center">
            <Spin size="large" />
            <div className="mt-4">
              <Text>Loading dashboard...</Text>
            </div>
          </div>
        </div>
      </Layout>
    );
  }

  // If no user data but authenticated, use localStorage data
  const displayUser = user || (typeof window !== 'undefined' ? AuthService.getUser() : null);

  const stats = {
    totalDocuments: documents.length,
    processedDocuments: documents.filter(doc => doc.status === 'processed').length,
    totalSize: documents.reduce((acc, doc) => acc + doc.file_size, 0),
  };

  return (
    <Layout>
      <div className="space-y-6">
        {/* Welcome Section */}
        <div className="text-center">
          <Title level={2}>Welcome back, {displayUser?.username}!</Title>
          <Text type="secondary">
            {displayUser?.role === 'admin' || displayUser?.role === 'super_admin' ? 'Administrator Dashboard' : 'Your Document Q&A Dashboard'}
          </Text>
        </div>

        {/* Statistics Cards */}
        <Row gutter={[16, 16]}>
          <Col xs={24} sm={12} md={6}>
            <Card>
              <Statistic
                title="Total Documents"
                value={stats.totalDocuments}
                prefix={<FileTextOutlined />}
                valueStyle={{ color: '#667eea' }}
              />
            </Card>
          </Col>
          <Col xs={24} sm={12} md={6}>
            <Card>
              <Statistic
                title="Processed"
                value={stats.processedDocuments}
                suffix={`/ ${stats.totalDocuments}`}
                prefix={<MessageOutlined />}
                valueStyle={{ color: '#52c41a' }}
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
                prefix={<InboxOutlined />}
                valueStyle={{ color: '#faad14' }}
              />
            </Card>
          </Col>
          <Col xs={24} sm={12} md={6}>
            <Card>
              <Statistic
                title="Your Role"
                value={
                  displayUser?.role === 'super_admin' ? 'Super Admin' :
                  displayUser?.role === 'admin' ? 'Admin' : 'User'
                }
                prefix={<UserOutlined />}
                valueStyle={{
                  color:
                    displayUser?.role === 'super_admin' ? '#ff4d4f' :
                    displayUser?.role === 'admin' ? '#722ed1' : '#1890ff'
                }}
              />
            </Card>
          </Col>
        </Row>

        {/* Quick Actions */}
        <Row gutter={[16, 16]}>
          <Col xs={24} lg={12}>
            <Card
              title="📁 Upload Documents"
              extra={<Button type="primary" icon={<UploadOutlined />}>Upload</Button>}
            >
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
          </Col>

          <Col xs={24} lg={12}>
            <Card
              title="💬 Start Chatting"
              extra={
                <Button
                  type="primary"
                  onClick={() => router.push('/chat')}
                  icon={<MessageOutlined />}
                >
                  Open Chat
                </Button>
              }
            >
              <div className="space-y-4">
                <Text>
                  Ask questions about your uploaded documents using AI-powered search.
                </Text>
                <div className="space-y-2">
                  <Text strong>Features:</Text>
                  <ul className="list-disc list-inside space-y-1 text-sm">
                    <li>Intelligent document search</li>
                    <li>Context-aware responses</li>
                    <li>Source citations</li>
                    <li>Multiple document support</li>
                  </ul>
                </div>
              </div>
            </Card>
          </Col>
        </Row>

        {/* Recent Documents */}
        <Card title="📄 Recent Documents">
          {documents.length > 0 ? (
            <div className="space-y-3">
              {documents.slice(0, 5).map((doc) => (
                <div
                  key={doc.id}
                  className="flex items-center justify-between p-3 border border-gray-200 rounded-lg hover:bg-gray-50"
                >
                  <div className="flex-1">
                    <Text strong>{doc.original_filename}</Text>
                    <br />
                    <Text type="secondary" className="text-sm">
                      {(doc.file_size / 1024).toFixed(1)} KB • {doc.status} • {new Date(doc.created_at).toLocaleDateString()}
                    </Text>
                  </div>
                  <div className="flex space-x-2">
                    <Button size="small" onClick={() => router.push('/chat')}>
                      Ask Questions
                    </Button>
                    <Button size="small" danger onClick={() => handleDeleteDocument(doc.id)}>
                      Delete
                    </Button>
                  </div>
                </div>
              ))}
              {documents.length > 5 && (
                <div className="text-center pt-4">
                  <Button onClick={() => router.push('/documents')}>
                    View All Documents ({documents.length})
                  </Button>
                </div>
              )}
            </div>
          ) : (
            <div className="text-center py-8">
              <FileTextOutlined style={{ fontSize: '48px', color: '#d9d9d9' }} />
              <div className="mt-4">
                <Title level={4} type="secondary">No documents uploaded yet</Title>
                <Text type="secondary">
                  Upload your first document using the upload area above to get started.
                </Text>
              </div>
            </div>
          )}
        </Card>
      </div>
    </Layout>
  );

  async function handleDeleteDocument(documentId: number) {
    try {
      await api.deleteDocument(documentId);
      message.success('Document deleted successfully');
      // Refresh documents list
      const updatedDocuments = await api.getDocuments();
      setDocuments(updatedDocuments);
    } catch (error) {
      message.error('Failed to delete document');
    }
  }
}