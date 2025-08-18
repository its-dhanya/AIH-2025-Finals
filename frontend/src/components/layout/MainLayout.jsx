// src/components/layout/MainLayout.jsx
import React from 'react';
import { Outlet } from 'react-router-dom'; // We'll install this next
import Header from './Header';
import Footer from './Footer';

const MainLayout = () => {
  return (
    <div className="flex flex-col min-h-screen bg-light-bg text-dark-panel">
      <Header />
      <main className="flex-grow">
        {/* Your page content will be rendered here */}
        <Outlet />
      </main>
      <Footer />
    </div>
  );
};

export default MainLayout;