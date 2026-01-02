import React, { useEffect, useState } from 'react';
import './ChatbotWidget.css';

declare global {
  interface Window {
    BookChatWidget: any;
  }
}

interface ChatbotWidgetProps {
  apiUrl?: string;
  bookId?: string;
  apiKey?: string;
}

const ChatbotWidget: React.FC<ChatbotWidgetProps> = ({
  apiUrl = 'http://localhost:8000/v1',
  bookId = 'humanoid-robotics-book',
  apiKey = 'test_api_key'  // Use the correct API key from the backend .env
}) => {
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [isWidgetLoaded, setIsWidgetLoaded] = useState(false);
  const [isWidgetInitialized, setIsWidgetInitialized] = useState(false);

  useEffect(() => {
    // Check if script is already loaded
    const existingScript = document.getElementById('chat-widget-script');
    if (existingScript) {
      setIsWidgetLoaded(true);
      return;
    }

    // Load the chat widget script dynamically
    const script = document.createElement('script');
    script.id = 'chat-widget-script';
    script.src = '/js/chat-widget.js';
    script.async = true;

    script.onload = () => {
      setIsWidgetLoaded(true);
      console.log('Chat widget script loaded');
    };

    script.onerror = () => {
      console.error('Failed to load chat widget script');
    };

    document.head.appendChild(script);

    return () => {
      // Cleanup script when component unmounts
      const scriptToRemove = document.getElementById('chat-widget-script');
      if (scriptToRemove) {
        document.head.removeChild(scriptToRemove);
      }
    };
  }, []);

  const toggleChat = () => {
    setIsChatOpen(!isChatOpen);
  };

  const initializeWidget = () => {
    if (window.BookChatWidget && !document.getElementById('book-chatbot-container')) {
      try {
        window.BookChatWidget.init({
          containerId: 'book-chatbot',
          apiUrl: apiUrl,
          bookId: bookId,
          apiKey: apiKey
        });
      } catch (error) {
        console.error('Error initializing chat widget:', error);
      }
    }
  };

  useEffect(() => {
    if (isChatOpen && isWidgetLoaded && !isWidgetInitialized) {
      // Wait for the modal to be rendered, then initialize the widget
      const timer = setTimeout(() => {
        initializeWidget();
        setIsWidgetInitialized(true);
      }, 100);

      return () => clearTimeout(timer);
    }
  }, [isChatOpen, isWidgetLoaded, isWidgetInitialized, apiUrl, bookId, apiKey]);

  // Reset initialization state when chat is closed
  useEffect(() => {
    if (!isChatOpen) {
      setIsWidgetInitialized(false);
    }
  }, [isChatOpen]);

  return (
    <>
      {isChatOpen && (
        <div className="chatbot-modal-overlay" onClick={() => setIsChatOpen(false)}>
          <div
            className="chatbot-modal-content"
            onClick={(e) => e.stopPropagation()}
          >
            <div id="book-chatbot" className="chatbot-container">
              {/* Placeholder while widget loads */}
              {!isWidgetInitialized && (
                <div className="chatbot-loading">
                  Loading chatbot...
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      <button
        className={`chatbot-float-button ${isChatOpen ? 'hidden' : ''}`}
        onClick={toggleChat}
        aria-label="Open chatbot"
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 24 24"
          fill="white"
          width="24px"
          height="24px"
        >
          <path d="M0 0h24v24H0z" fill="none"/>
          <path d="M20 2H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h4v3c0 .6.4 1 1 1 .2 0 .5-.1.7-.3L14.4 18H20c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2z"/>
        </svg>
      </button>
    </>
  );
};

export default ChatbotWidget;