import React, { useState } from 'react';
import Layout from './components/Layout';
import { api } from './api/client';
import { Bot, TrendingUp, ShieldAlert, BarChart3, Users, DollarSign } from 'lucide-react';

export default function App() {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    try {
      const data = await api.ask(query);
      setResponse(data);
    } catch (err) {
      console.error(err);
      setResponse({ error: 'Failed to process query' });
    } finally {
      setLoading(false);
    }
  };

  return (
    <Layout>
      <div className="max-w-4xl mx-auto space-y-8 animate-fade-in">
        
        {/* Header Section */}
        <div className="text-center space-y-4 py-8">
          <div className="inline-flex items-center justify-center p-3 bg-blue-500/10 rounded-2xl mb-4">
            <Bot className="w-10 h-10 text-blue-400" />
          </div>
          <h1 className="text-4xl md:text-6xl font-extrabold tracking-tight bg-gradient-to-r from-blue-400 via-indigo-400 to-purple-400 text-transparent bg-clip-text">
            NBA Agentic AI
          </h1>
          <p className="text-lg text-gray-400 max-w-2xl mx-auto">
            Your personal front office. Ask any trade, roster, or salary question and let our 10 specialized agents analyze it.
          </p>
        </div>

        {/* Input Section */}
        <form onSubmit={handleSubmit} className="relative group">
          <div className="absolute -inset-1 bg-gradient-to-r from-blue-500 to-purple-600 rounded-2xl blur opacity-25 group-hover:opacity-40 transition duration-1000 group-hover:duration-200"></div>
          <div className="relative glass-panel p-2 flex items-center">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="e.g., Analyze trading LeBron James to the Celtics for Jayson Tatum..."
              className="flex-1 bg-transparent border-none focus:ring-0 text-lg px-4 py-3 placeholder-gray-500 outline-none"
              disabled={loading}
            />
            <button 
              type="submit" 
              disabled={loading || !query.trim()}
              className="btn-primary px-8 ml-2 flex items-center justify-center"
            >
              {loading ? (
                <div className="w-6 h-6 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
              ) : (
                'Analyze'
              )}
            </button>
          </div>
        </form>

        {/* Features Grid (Shown when no response) */}
        {!response && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-12">
            <FeatureCard 
              icon={<TrendingUp />} 
              title="Trade Evaluation" 
              desc="Comprehensive analysis of multi-team trades."
            />
            <FeatureCard 
              icon={<DollarSign />} 
              title="Salary Cap Rules" 
              desc="Strict 125% rule matching and filler suggestions."
            />
            <FeatureCard 
              icon={<ShieldAlert />} 
              title="Injury Risk" 
              desc="Predictive injury models based on workload."
            />
          </div>
        )}

        {/* Response Section */}
        {response && (
          <div className="animate-slide-up space-y-6">
            {response.error ? (
              <div className="glass-panel p-6 border-red-500/30 bg-red-500/5">
                <p className="text-red-400 text-lg">{response.error}</p>
              </div>
            ) : (
              <>
                <div className="glass-panel p-8 prose prose-invert max-w-none prose-headings:text-gray-100 prose-a:text-blue-400">
                  <div dangerouslySetInnerHTML={{ __html: renderMarkdown(response.response) }} />
                </div>
                
                {response.tool_calls && response.tool_calls.length > 0 && (
                  <div className="glass-panel p-6">
                    <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4 flex items-center gap-2">
                      <BarChart3 className="w-4 h-4" />
                      Agent Reasoning Trace
                    </h3>
                    <div className="flex flex-wrap gap-2">
                      {response.tool_calls.map((call, idx) => (
                        <span key={idx} className="px-3 py-1 bg-gray-800 border border-gray-700 rounded-full text-xs font-mono text-gray-300">
                          {call.tool}()
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        )}

      </div>
    </Layout>
  );
}

function FeatureCard({ icon, title, desc }) {
  return (
    <div className="glass-panel glass-panel-hover p-6 text-center space-y-4">
      <div className="inline-flex p-3 bg-gray-800 rounded-xl text-blue-400">
        {icon}
      </div>
      <h3 className="text-lg font-semibold text-gray-100">{title}</h3>
      <p className="text-sm text-gray-400 leading-relaxed">{desc}</p>
    </div>
  );
}

// Simple markdown renderer for the response
function renderMarkdown(text) {
  if (!text) return '';
  return text
    .replace(/^## (.*$)/gim, '<h2 class="text-2xl font-bold mt-8 mb-4">$1</h2>')
    .replace(/^### (.*$)/gim, '<h3 class="text-xl font-semibold mt-6 mb-3">$1</h3>')
    .replace(/\*\*(.*)\*\*/gim, '<strong>$1</strong>')
    .replace(/\*(.*)\*/gim, '<em>$1</em>')
    .replace(/^- (.*$)/gim, '<li class="ml-4 mb-1">$1</li>')
    .replace(/\n/g, '<br/>')
    .replace(/(<li.*<\/li>)/gim, '<ul class="list-disc mb-4">$1</ul>');
}
