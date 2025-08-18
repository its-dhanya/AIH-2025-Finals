// src/components/layout/Header.jsx
import React from 'react';

const Header = () => {
  return (
    <header className="bg-white shadow-sm sticky top-0 z-50">
      <nav className="container mx-auto px-6 py-3 flex justify-between items-center">
        <div className="text-2xl font-bold text-primary">
          PDF Connect
        </div>
        <div>
          {/* We'll make this a real component later */}
          <span className="text-sm font-medium p-2">Analyst / Executive</span>
        </div>
      </nav>
    </header>
  );
};

export default Header;