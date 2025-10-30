'use client';

import React, { useState, useEffect } from 'react';
import {
  Layout as AntLayout,
  Menu,
  Button,
  Avatar,
  Dropdown,
  Badge,
  Typography,
  Space,
  Drawer,
  ConfigProvider,
} from 'antd';
import {
  MenuFoldOutlined,
  MenuUnfoldOutlined,
  UserOutlined,
  LogoutOutlined,
  DashboardOutlined,
  FileTextOutlined,
  MessageOutlined,
  SettingOutlined,
  TeamOutlined,
  FileExcelOutlined,
} from '@ant-design/icons';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { AuthService } from '@/lib/auth';
import { api } from '@/lib/api';
import type { User } from '@/types';

const { Header, Sider, Content } = AntLayout;
const { Text } = Typography;

interface LayoutProps {
  children: React.ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  const [collapsed, setCollapsed] = useState(false);
  const [mobileMenuVisible, setMobileMenuVisible] = useState(false);
  const [user, setUser] = useState<User | null>(null);
  const [loggingOut, setLoggingOut] = useState(false);
  const [companyName, setCompanyName] = useState<string>('');
  const [companyLogo, setCompanyLogo] = useState<string>('');
  const [primaryColor, setPrimaryColor] = useState<string>('#1890ff');
  const pathname = usePathname();

  useEffect(() => {
    // Load user data on client side only
    const userData = AuthService.getUser();
    setUser(userData);

    // Force re-render when user data changes (for role updates)
    const handleStorageChange = () => {
      const updatedUserData = AuthService.getUser();
      setUser(updatedUserData);
    };

    // Listen for storage changes (in case user data is updated elsewhere)
    window.addEventListener('storage', handleStorageChange);

    // Load branding and theme
    const loadBranding = async () => {
      try {
        const branding = await api.getCompanyBranding();
        setCompanyName(branding.company_name);
        setCompanyLogo(branding.logo_url);
      } catch (error) {
        console.error('Failed to load branding:', error);
      }
    };
    loadBranding();

    // Load theme color
    setPrimaryColor(localStorage.getItem('primaryColor') || '#1890ff');

    // Listen for theme changes
    const handleThemeChange = () => {
      setPrimaryColor(localStorage.getItem('primaryColor') || '#1890ff');
    };
    window.addEventListener('themeChange', handleThemeChange);

    return () => {
      window.removeEventListener('storage', handleStorageChange);
      window.removeEventListener('themeChange', handleThemeChange);
    };
  }, []);

  useEffect(() => {
    document.documentElement.style.setProperty('--primary-color', primaryColor);
  }, [primaryColor]);

  const menuItems = [
    {
      key: '/dashboard',
      icon: <DashboardOutlined />,
      label: <Link href="/dashboard">Dashboard</Link>,
    },
    {
      key: '/documents',
      icon: <FileTextOutlined />,
      label: <Link href="/documents">Documents</Link>,
    },
    {
      key: '/chat',
      icon: <MessageOutlined />,
      label: <Link href="/chat">Chat</Link>,
    },
    {
      key: '/question-export',
      icon: <FileExcelOutlined />,
      label: <Link href="/question-export">Export Questions</Link>,
    },
  ];

  // Add admin menu items if user is admin or super_admin
  if (user?.role === 'admin' || user?.role === 'super_admin') {
    menuItems.push(
      {
        key: '/admin/users',
        icon: <TeamOutlined />,
        label: <Link href="/admin/users">Users</Link>,
      }
    );
    
    // Only super_admin can access settings
    if (user?.role === 'super_admin') {
      menuItems.push(
        {
          key: '/admin/settings',
          icon: <SettingOutlined />,
          label: <Link href="/admin/settings">Settings</Link>,
        }
      );
    }
  }

  const userMenuItems = [
    {
      key: 'profile',
      label: 'Profile',
      icon: <UserOutlined />,
    },
    {
      key: 'logout',
      label: 'Logout',
      icon: <LogoutOutlined />,
      onClick: async () => {
        if (loggingOut) return; // Prevent multiple clicks

        setLoggingOut(true);
        try {
          await AuthService.logout();
          console.log('Logout successful');
          // Use window.location for full page reload to ensure clean state
          window.location.href = '/login';
        } catch (error) {
          console.error('Logout error:', error);
          // Even if logout fails, force redirect to login
          window.location.href = '/login';
        } finally {
          setLoggingOut(false);
        }
      },
    },
  ];

  const SidebarContent = () => (
    <div className="flex flex-col h-full">
      {/* Logo/Brand */}
      <div className="flex items-center justify-center h-16 px-4 border-b border-gray-200">
        <div className="flex items-center">
          {companyLogo ? (
            <img src={companyLogo} alt="Company Logo" className="h-6 mr-2" />
          ) : (
            <span className="text-lg mr-2">📚</span>
          )}
          <Text strong className="text-lg">
            {companyName || 'Doc Q&A'}
          </Text>
        </div>
      </div>

      {/* Navigation Menu */}
      <Menu
        mode="inline"
        selectedKeys={[pathname]}
        items={menuItems}
        className="flex-1 border-r-0"
      />
    </div>
  );

  return (
    <ConfigProvider theme={{ token: { colorPrimary: primaryColor } }}>
      <AntLayout className="min-h-screen">
      {/* Desktop Sidebar */}
      <Sider
        trigger={null}
        collapsible
        collapsed={collapsed}
        className="hidden md:block"
        width={280}
      >
        <SidebarContent />
      </Sider>

      {/* Mobile Sidebar */}
      <Drawer
        title={
          <div className="flex items-center">
            {companyLogo ? (
              <img src={companyLogo} alt="Company Logo" className="h-6 mr-2" />
            ) : (
              <span className="text-lg mr-2">📚</span>
            )}
            <span>{companyName || 'Doc Q&A'}</span>
          </div>
        }
        placement="left"
        onClose={() => setMobileMenuVisible(false)}
        open={mobileMenuVisible}
        width={280}
        styles={{ body: { padding: 0 } }}
      >
        <SidebarContent />
      </Drawer>

      <AntLayout>
        {/* Header */}
        <Header className="px-4 text-white flex items-center justify-between" style={{ backgroundColor: primaryColor, borderBottom: `1px solid ${primaryColor}40` }}>
          <Space>
            {/* Mobile menu button */}
            <Button
              type="text"
              icon={collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
              onClick={() => setMobileMenuVisible(true)}
              className="md:hidden"
            />

            {/* Desktop collapse button */}
            <Button
              type="text"
              icon={collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
              onClick={() => setCollapsed(!collapsed)}
              className="hidden md:block"
            />
          </Space>

          <Space>
            {/* User dropdown */}
            <Dropdown
              menu={{
                items: userMenuItems,
              }}
              placement="bottomRight"
            >
              <Button type="text" className="flex items-center space-x-2">
                <Avatar icon={<UserOutlined />} />
                <span className="hidden sm:block">
                  {loggingOut ? 'Logging out...' : user?.username}
                </span>
                {(user?.role === 'admin' || user?.role === 'super_admin') && (
                  <Badge
                    count={user?.role === 'super_admin' ? 'Super Admin' : 'Admin'}
                    style={{
                      backgroundColor: user?.role === 'super_admin' ? '#ff4d4f' : '#667eea'
                    }}
                  />
                )}
              </Button>
            </Dropdown>
          </Space>
        </Header>

        {/* Main Content */}
        <Content className="p-6 bg-gray-50">
          <div className="max-w-7xl mx-auto">
            {children}
          </div>
        </Content>
      </AntLayout>
    </AntLayout>
    </ConfigProvider>
  );
};

export default Layout;