import React from 'react';

export default function Layout({ children }) {
  return (
    <div className="min-h-screen relative overflow-hidden bg-grid-pattern">
      {/* Ambient background glows */}
      <div className="absolute top-0 left-1/4 w-96 h-96 bg-blue-500/10 rounded-full blur-[120px] pointer-events-none"></div>
      <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-purple-500/10 rounded-full blur-[120px] pointer-events-none"></div>

      {/* Navbar */}
      <nav className="sticky top-0 z-50 border-b border-gray-800 bg-[#0B0F19]/80 backdrop-blur-xl">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center">
              <span className="text-white font-bold text-sm">AI</span>
            </div>
            <span className="font-semibold tracking-wide text-gray-100">NBA Agentic AI</span>
          </div>
          <div className="flex items-center gap-6 text-sm font-medium text-gray-400">
            <a href="#" className="text-gray-100 hover:text-white transition-colors">Simulator</a>
            <a href="#" className="hover:text-white transition-colors">Roster Builder</a>
            <a href="#" className="hover:text-white transition-colors">Draft</a>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-12 relative z-10">
        {children}
      </main>
    </div>
  );
}
