import React, { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import ChatHeader from "./ChatHeader";
import ChatMessages from "./ChatMessages";
import ChatInput from "./ChatInput";

const BACKEND_URL = "http://127.0.0.1:8000";

const Chatbot = () => {
  const [chatExpanded, setChatExpanded] = useState(false);
  const [messages, setMessages] = useState([]);
  const [conversationId, setConversationId] = useState(null);
  const [prompt, setPrompt] = useState("");
  const [loading, setLoading] = useState(false);
  const [rateLimitInfo, setRateLimitInfo] = useState(null);
  const [isRateLimited, setIsRateLimited] = useState(false);
  const chatContainerRef = useRef(null);

  const toggleChat = () => {
    setChatExpanded((prev) => !prev);
    document.body.style.overflow = chatExpanded ? "" : "hidden";
  };

  // Fetch rate limit info when component mounts or chat is expanded
  const fetchRateLimitInfo = async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/rate-limit-info`, {
        method: "GET",
        headers: { "Content-Type": "application/json" },
      });
      
      if (response.ok) {
        const data = await response.json();
        setRateLimitInfo(data.rate_limit);
        setIsRateLimited(data.rate_limit.remaining <= 0);
      }
    } catch (error) {
      console.error("Failed to fetch rate limit info:", error);
    }
  };

  // Update rate limit info from response headers
  const updateRateLimitFromHeaders = (headers) => {
    const limit = headers.get('X-RateLimit-Limit');
    const remaining = headers.get('X-RateLimit-Remaining');
    const reset = headers.get('X-RateLimit-Reset');
    
    if (limit && remaining && reset) {
      setRateLimitInfo({
        limit: parseInt(limit),
        remaining: parseInt(remaining),
        reset_time: reset,
        used: parseInt(limit) - parseInt(remaining)
      });
      setIsRateLimited(parseInt(remaining) <= 0);
    }
  };

  const sendMessage = async () => {
    if (!prompt.trim() || loading || isRateLimited) return;

    const newMessages = [...messages, { role: "user", content: prompt }];
    setMessages(newMessages);
    setPrompt("");
    setLoading(true);

    try {
      const payload = {
        question: prompt,
        conversation_id: conversationId,
        messages: newMessages,
      };

      const response = await fetch(`${BACKEND_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      // Update rate limit info from headers
      updateRateLimitFromHeaders(response.headers);

      if (response.status === 429) {
        // Rate limit exceeded
        const errorData = await response.json();
        setMessages([
          ...newMessages,
          {
            role: "assistant",
            content: `⚠️ ${errorData.message}\n\nYou have used all your daily requests. Please try again tomorrow.\n\nRemaining requests: ${errorData.rate_limit_info.remaining}/${errorData.rate_limit_info.limit}`,
            isError: true,
            isRateLimit: true,
          },
        ]);
        setIsRateLimited(true);
        return;
      }

      const data = await response.json();

      setMessages([
        ...newMessages,
        {
          role: "assistant",
          content: data.answer || "No response available.",
          source_link_metadata: data.source_link_metadata || null,
        },
      ]);
      setConversationId(data.conversation_id);
    } catch (error) {
      setMessages([
        ...newMessages,
        {
          role: "assistant",
          content: `Error: ${error.message}`,
          isError: true,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      sendMessage();
    }
    if (e.key === "Escape" && chatExpanded) {
      toggleChat();
    }
  };

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages]);

  useEffect(() => {
    if (chatExpanded) {
      fetchRateLimitInfo();
    }
  }, [chatExpanded]);

  const formatResetTime = (resetTime) => {
    if (!resetTime) return "Unknown";
    const resetDate = new Date(resetTime);
    const now = new Date();
    const diffHours = Math.ceil((resetDate - now) / (1000 * 60 * 60));
    
    if (diffHours <= 0) return "Soon";
    if (diffHours < 24) return `${diffHours} hour${diffHours > 1 ? 's' : ''}`;
    return resetDate.toLocaleDateString();
  };

  return (
    <div className="fixed bottom-4 right-4 z-50">
      {/* Chat Toggle Button */}
      <motion.button
        onClick={toggleChat}
        className="bg-blue-600 hover:bg-blue-700 text-white rounded-full p-4 shadow-lg transition-colors duration-200"
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          className="h-6 w-6"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
          />
        </svg>
      </motion.button>

      {/* Chat Overlay */}
      <AnimatePresence>
        {chatExpanded && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4"
            onClick={toggleChat}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="bg-white rounded-lg shadow-xl w-full max-w-md h-96 flex flex-col"
              onClick={(e) => e.stopPropagation()}
            >
              <ChatHeader onClose={toggleChat} />
              
              {/* Rate Limit Info */}
              {rateLimitInfo && (
                <div className="px-4 py-2 bg-gray-50 border-b text-sm">
                  <div className="flex justify-between items-center">
                    <span className={`font-medium ${isRateLimited ? 'text-red-600' : 'text-green-600'}`}>
                      Requests: {rateLimitInfo.remaining}/{rateLimitInfo.limit}
                    </span>
                    {isRateLimited && (
                      <span className="text-red-500 text-xs">
                        Resets in: {formatResetTime(rateLimitInfo.reset_time)}
                      </span>
                    )}
                  </div>
                  {isRateLimited && (
                    <div className="text-red-500 text-xs mt-1">
                      Daily limit reached. Please try again tomorrow.
                    </div>
                  )}
                </div>
              )}

              <ChatMessages
                messages={messages}
                loading={loading}
                chatContainerRef={chatContainerRef}
              />
              
              <ChatInput
                prompt={prompt}
                setPrompt={setPrompt}
                sendMessage={sendMessage}
                handleKeyDown={handleKeyDown}
                loading={loading}
                disabled={isRateLimited}
                placeholder={isRateLimited ? "Daily limit reached" : "Type your message..."}
              />
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default Chatbot;