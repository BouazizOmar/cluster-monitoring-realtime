import React from 'react';
import { MessageSquare } from 'lucide-react';

interface HeaderProps {
  title?: string;
}

const Header: React.FC<HeaderProps> = ({ 
  title = "Analytics Assistant" 
}) => {
  return (
    <header className="bg-white border-b border-gray-200 py-4 px-6 shadow-sm">
      <div className="max-w-4xl mx-auto flex justify-between items-center">
        <div className="flex items-center space-x-2">
          <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center">
            <MessageSquare className="h-4 w-4 text-white" />
          </div>
          <div className="text-blue-600 font-bold text-xl">{title}</div>
        </div>
        
        <div className="flex items-center space-x-4">
          <button className="text-gray-500 hover:text-gray-700 text-sm font-medium">
            Documentation
          </button>
          <button className="text-gray-500 hover:text-gray-700 text-sm font-medium">
            Settings
          </button>
        </div>
      </div>
    </header>
  );
};

export default Header;