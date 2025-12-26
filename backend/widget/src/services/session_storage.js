/**
 * Session Storage Service for the Book Chatbot Widget
 * Handles client-side session management using localStorage
 */

import { SessionModel } from '../models/session.js';

export class SessionStorageService {
    constructor(options = {}) {
        this.storageKeyPrefix = options.storageKeyPrefix || 'book-chat-session';
        this.sessionTimeout = options.sessionTimeout || 24 * 60 * 60 * 1000; // 24 hours in milliseconds
        this.maxSessions = options.maxSessions || 10; // Maximum number of stored sessions
    }

    generateSessionKey(sessionId, bookId) {
        return `${this.storageKeyPrefix}-${bookId}-${sessionId}`;
    }

    async createSession(bookId) {
        const sessionId = this.generateSessionId();
        const session = new SessionModel(sessionId, bookId);

        await this.saveSession(session);

        return session;
    }

    async saveSession(session) {
        try {
            const sessionKey = this.generateSessionKey(session.id, session.bookId);
            const sessionData = session.toJSON();

            // Store the session in localStorage
            localStorage.setItem(sessionKey, JSON.stringify(sessionData));

            // Update the list of active sessions
            await this.updateActiveSessions(session.bookId, session.id);

            return true;
        } catch (error) {
            console.error('Error saving session:', error);
            return false;
        }
    }

    async loadSession(sessionId, bookId) {
        try {
            const sessionKey = this.generateSessionKey(sessionId, bookId);
            const sessionData = localStorage.getItem(sessionKey);

            if (!sessionData) {
                return null;
            }

            const parsedData = JSON.parse(sessionData);
            const session = SessionModel.fromJSON(parsedData);

            // Check if session is expired
            if (session.isExpired()) {
                await this.removeSession(sessionId, bookId);
                return null;
            }

            // Update last activity
            session.lastActivity = new Date().toISOString();
            await this.saveSession(session);

            return session;
        } catch (error) {
            console.error('Error loading session:', error);
            return null;
        }
    }

    async loadOrCreateSession(bookId, sessionId = null) {
        let session = null;

        if (sessionId) {
            session = await this.loadSession(sessionId, bookId);
        }

        if (!session) {
            session = await this.createSession(bookId);
        }

        return session;
    }

    async removeSession(sessionId, bookId) {
        try {
            const sessionKey = this.generateSessionKey(sessionId, bookId);
            localStorage.removeItem(sessionKey);

            // Remove from active sessions list
            await this.removeFromActiveSessions(bookId, sessionId);

            return true;
        } catch (error) {
            console.error('Error removing session:', error);
            return false;
        }
    }

    async clearBookSessions(bookId) {
        try {
            // Get all active sessions for this book
            const activeSessions = await this.getActiveSessions(bookId);

            // Remove each session
            for (const sessionId of activeSessions) {
                await this.removeSession(sessionId, bookId);
            }

            // Clear the active sessions list
            await this.clearActiveSessions(bookId);

            return true;
        } catch (error) {
            console.error('Error clearing book sessions:', error);
            return false;
        }
    }

    async getAllSessions(bookId) {
        try {
            const activeSessions = await this.getActiveSessions(bookId);
            const sessions = [];

            for (const sessionId of activeSessions) {
                const session = await this.loadSession(sessionId, bookId);
                if (session && !session.isExpired()) {
                    sessions.push(session);
                }
            }

            return sessions;
        } catch (error) {
            console.error('Error getting all sessions:', error);
            return [];
        }
    }

    async updateActiveSessions(bookId, sessionId) {
        try {
            const activeSessionsKey = `${this.storageKeyPrefix}-active-${bookId}`;
            let activeSessions = JSON.parse(localStorage.getItem(activeSessionsKey) || '[]');

            // Add the new session ID if not already present
            if (!activeSessions.includes(sessionId)) {
                activeSessions.unshift(sessionId); // Add to the beginning to maintain recency order
            }

            // Keep only the most recent sessions up to maxSessions
            if (activeSessions.length > this.maxSessions) {
                // Remove the oldest sessions
                const sessionsToRemove = activeSessions.slice(this.maxSessions);
                activeSessions = activeSessions.slice(0, this.maxSessions);

                // Actually remove the old sessions from storage
                for (const oldSessionId of sessionsToRemove) {
                    const oldSessionKey = this.generateSessionKey(oldSessionId, bookId);
                    localStorage.removeItem(oldSessionKey);
                }
            }

            localStorage.setItem(activeSessionsKey, JSON.stringify(activeSessions));
        } catch (error) {
            console.error('Error updating active sessions:', error);
        }
    }

    async removeFromActiveSessions(bookId, sessionId) {
        try {
            const activeSessionsKey = `${this.storageKeyPrefix}-active-${bookId}`;
            let activeSessions = JSON.parse(localStorage.getItem(activeSessionsKey) || '[]');

            // Remove the session ID
            activeSessions = activeSessions.filter(id => id !== sessionId);

            localStorage.setItem(activeSessionsKey, JSON.stringify(activeSessions));
        } catch (error) {
            console.error('Error removing from active sessions:', error);
        }
    }

    async getActiveSessions(bookId) {
        try {
            const activeSessionsKey = `${this.storageKeyPrefix}-active-${bookId}`;
            return JSON.parse(localStorage.getItem(activeSessionsKey) || '[]');
        } catch (error) {
            console.error('Error getting active sessions:', error);
            return [];
        }
    }

    async clearActiveSessions(bookId) {
        try {
            const activeSessionsKey = `${this.storageKeyPrefix}-active-${bookId}`;
            localStorage.removeItem(activeSessionsKey);
        } catch (error) {
            console.error('Error clearing active sessions:', error);
        }
    }

    generateSessionId() {
        // Generate a unique session ID
        return `sess_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    async validateSession(sessionId, bookId) {
        try {
            const session = await this.loadSession(sessionId, bookId);
            return session !== null && !session.isExpired();
        } catch (error) {
            console.error('Error validating session:', error);
            return false;
        }
    }

    async cleanupExpiredSessions(bookId) {
        try {
            const activeSessions = await this.getActiveSessions(bookId);
            const validSessions = [];

            for (const sessionId of activeSessions) {
                const session = await this.loadSession(sessionId, bookId);

                if (!session || session.isExpired()) {
                    // Session is expired, remove it
                    await this.removeSession(sessionId, bookId);
                } else {
                    // Session is still valid
                    validSessions.push(sessionId);
                }
            }

            return validSessions.length;
        } catch (error) {
            console.error('Error cleaning up expired sessions:', error);
            return 0;
        }
    }

    // Privacy compliance method - removes all personal data
    async ensurePrivacyCompliance() {
        try {
            // Get all keys in localStorage that belong to this application
            const keysToRemove = [];
            for (let i = 0; i < localStorage.length; i++) {
                const key = localStorage.key(i);
                if (key && key.startsWith(this.storageKeyPrefix)) {
                    keysToRemove.push(key);
                }
            }

            // Remove all matching keys
            for (const key of keysToRemove) {
                localStorage.removeItem(key);
            }

            return {
                status: 'success',
                keysRemoved: keysToRemove.length,
                message: 'Privacy compliance ensured - all session data removed'
            };
        } catch (error) {
            console.error('Error ensuring privacy compliance:', error);
            return {
                status: 'error',
                message: error.message
            };
        }
    }
}