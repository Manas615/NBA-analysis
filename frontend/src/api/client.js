/**
 * API Client for interacting with the NBA Agentic AI backend.
 */

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

class ApiClient {
  constructor() {
    this.sessionId = localStorage.getItem('nba_session_id') || null;
  }

  async _fetch(endpoint, options = {}) {
    const url = `${BASE_URL}${endpoint}`;
    
    const headers = {
      'Content-Type': 'application/json',
      ...options.headers,
    };

    try {
      const response = await fetch(url, {
        ...options,
        headers,
      });

      if (!response.ok) {
        throw new Error(`API Error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      
      // Save session ID for follow-up queries
      if (data.session_id && this.sessionId !== data.session_id) {
        this.sessionId = data.session_id;
        localStorage.setItem('nba_session_id', data.session_id);
      }

      return data;
    } catch (error) {
      console.error('Fetch error:', error);
      throw error;
    }
  }

  async ask(query) {
    return this._fetch('/ask', {
      method: 'POST',
      body: JSON.stringify({
        query,
        session_id: this.sessionId,
      }),
    });
  }

  async analyzeTrade(teamA, playerA, teamB, playerB) {
    return this._fetch('/trade', {
      method: 'POST',
      body: JSON.stringify({
        team_a: teamA,
        player_a: playerA,
        team_b: teamB,
        player_b: playerB,
        session_id: this.sessionId,
      }),
    });
  }
}

export const api = new ApiClient();
