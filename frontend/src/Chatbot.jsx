import React, { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";

// const BACKEND_URL = "https://margdarshak-backend.onrender.com"|| "";
// const BACKEND_URL = "http://15.207.109.149:8000";
const BACKEND_URL = "http://127.0.0.1:8000"; // For local development
// const BACKEND_URL = "https://margdarshak.tech";

const Chatbot = () => {
  const [chatExpanded, setChatExpanded] = useState(false);
  const [messages, setMessages] = useState([]);
  const [conversationId, setConversationId] = useState(null);
  const [prompt, setPrompt] = useState("");
  const [loading, setLoading] = useState(false);
  const [userIP, setUserIP] = useState(null);
  const [rateLimitInfo, setRateLimitInfo] = useState(null);
  const [rateLimitError, setRateLimitError] = useState(null);
  const chatContainerRef = useRef(null);

  // Function to get user's IP address
  async function getUserIP() {
    try {
      // Method 1: Use ipify service (most reliable)
      const response = await fetch("https://api.ipify.org?format=json");
      const data = await response.json();
      return data.ip;
    } catch (error) {
      console.warn("Failed to get IP from ipify, trying alternative...", error);

      try {
        // Method 2: Alternative service
        const response = await fetch("https://api.ip.sb/ip");
        const ip = await response.text();
        return ip.trim();
      } catch (error2) {
        console.warn("Failed to get IP from alternative service", error2);

        try {
          // Method 3: Another alternative
          const response = await fetch("https://ipapi.co/ip/");
          const ip = await response.text();
          return ip.trim();
        } catch (error3) {
          console.error("All IP detection methods failed", error3);
          return null;
        }
      }
    }
  }

  // Function to check rate limit status
  async function checkRateLimit(ip, apiBaseUrl = BACKEND_URL) {
    try {
      const response = await fetch(`${apiBaseUrl}/rate-limit-status/${ip}`);
      const data = await response.json();
      return data;
    } catch (error) {
      console.error("Failed to check rate limit status:", error);
      return null;
    }
  }

  // Function to send chat message with rate limiting
  async function sendChatMessage(messageData, apiBaseUrl = BACKEND_URL) {
    try {
      // Add IP to the request payload
      const requestData = {
        ...messageData,
        user_ip: userIP,
      };

      const response = await fetch(`${apiBaseUrl}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestData),
      });

      const data = await response.json();

      // Handle rate limit exceeded
      if (response.status === 429) {
        return {
          success: false,
          error: "rate_limit_exceeded",
          message: data.detail?.message || "Rate limit exceeded",
          quota_limit: data.detail?.quota_limit || 5,
          reset_time: data.detail?.reset_time,
        };
      }

      // Handle other errors
      if (!response.ok) {
        return {
          success: false,
          error: "api_error",
          message: data.detail || "An error occurred",
        };
      }

      // Success - the rate_limit_info is now included in the response
      return {
        success: true,
        data: data,
        rate_limit_info: data.rate_limit_info, // This contains the updated counts
      };
    } catch (error) {
      console.error("Chat request failed:", error);
      return {
        success: false,
        error: "network_error",
        message: "Failed to send message. Please check your connection.",
      };
    }
  }

  // Get user IP on component mount
  useEffect(() => {
    getUserIP().then((ip) => {
      setUserIP(ip);
      console.log("User IP set:", ip);

      // Check initial rate limit status
      if (ip) {
        checkRateLimit(ip, BACKEND_URL).then((status) => {
          if (status) {
            setRateLimitInfo(status);
          }
        });
      }
    });
  }, []);

  const toggleChat = () => {
    setChatExpanded((prev) => !prev);

    // Handle body overflow when chat is expanded/collapsed
    if (!chatExpanded) {
      // Expanding - hide body overflow
      document.body.style.overflow = "hidden";
    } else {
      // Collapsing - restore body overflow
      document.body.style.overflow = "";
    }
  };

  const sendMessage = async () => {
    if (!prompt.trim() || loading) return;

    const newMessages = [...messages, { role: "user", content: prompt }];
    setMessages(newMessages);
    setPrompt("");
    setLoading(true);
    setRateLimitError(null);

    try {
      const payload = {
        question: prompt,
        conversation_id: conversationId,
        messages: newMessages,
      };

      console.log("payload", payload);

      // Use the rate-limited send function
      const result = await sendChatMessage(payload, BACKEND_URL);

      if (result.success) {
        const data = result.data;
        console.log("Response:", data);

        // Include source_links and source_link_metadata in the assistant message
        setMessages([
          ...newMessages,
          {
            role: "assistant",
            content: data.answer || "No response available.",
            source_link_metadata: data.source_link_metadata || null,
          },
        ]);
        setConversationId(data.conversation_id);

        // Update rate limit info
        if (result.rate_limit_info) {
          setRateLimitInfo(result.rate_limit_info);
        }
      } else {
        // Handle errors
        if (result.error === "rate_limit_exceeded") {
          setRateLimitError(result.message);
          setMessages([
            ...newMessages,
            {
              role: "assistant",
              content: `Rate limit exceeded: ${result.message}`,
              isError: true,
            },
          ]);
        } else {
          setMessages([
            ...newMessages,
            {
              role: "assistant",
              content: `Error: ${result.message}`,
              isError: true,
            },
          ]);
        }
      }
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
    // Handle Escape key to close chat
    if (e.key === "Escape" && chatExpanded) {
      toggleChat();
    }
  };

  const checkUserRateLimit = async () => {
    if (!userIP) return;

    const status = await checkRateLimit(userIP, BACKEND_URL);
    console.log("Rate limit status:", status);
    if (status && status.requests_remaining === 0) {
      setRateLimitError("Daily quota exceeded. Please try again tomorrow.");
    }
    if (status) {
      setRateLimitInfo(status);
    }
  };

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop =
        chatContainerRef.current.scrollHeight;
    }
  }, [messages]);

  useEffect(() => {
    const link = document.createElement("link");
    link.href =
      "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap";
    link.rel = "stylesheet";
    document.head.appendChild(link);

    // Add global event listeners
    const handleGlobalKeyDown = (e) => {
      if (e.key === "Escape" && chatExpanded) {
        toggleChat();
      }
    };

    document.addEventListener("keydown", handleGlobalKeyDown);

    return () => {
      if (document.head.contains(link)) {
        document.head.removeChild(link);
      }
      document.removeEventListener("keydown", handleGlobalKeyDown);
      // Cleanup: restore body overflow when component unmounts
      document.body.style.overflow = "";
    };
  }, [chatExpanded]);

  const fullScreenVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        duration: 0.3,
      },
    },
    exit: {
      opacity: 0,
      transition: {
        duration: 0.2,
      },
    },
  };

  const messageVariants = {
    hidden: { opacity: 0, y: 10 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        type: "spring",
        damping: 20,
        stiffness: 300,
      },
    },
  };

  const inputGroupVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        delay: 0.2,
        duration: 0.3,
      },
    },
  };

  const typingIndicatorVariants = {
    initial: { scale: 0.8, opacity: 0.5 },
    animate: {
      scale: [0.8, 1, 0.8],
      opacity: [0.5, 1, 0.5],
      transition: {
        duration: 1,
        repeat: Infinity,
        repeatType: "loop",
      },
    },
  };

  // Rate Limit Status Component

  // Updated Rate Limit Status Component
  const RateLimitStatus = () => {
    if (!rateLimitInfo) return null;

    const used = rateLimitInfo.requests_used || 0;
    const remaining = rateLimitInfo.requests_remaining || 0;
    const total = rateLimitInfo.quota_limit || 5;
    const isLimitReached = remaining === 0;

    return (
      <motion.div
        className={`px-4 py-2 rounded-lg text-sm font-medium ${
          isLimitReached
            ? "bg-red-100 text-red-800 border border-red-200"
            : remaining <= 2
            ? "bg-yellow-100 text-yellow-800 border border-yellow-200"
            : "bg-green-100 text-green-800 border border-green-200"
        }`}
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
      >
        <div className="flex items-center justify-between gap-4">
          <span>
            📊 Daily Quota: {used}/{total} used
            {remaining > 0 ? ` (${remaining} remaining)` : " (Limit reached)"}
          </span>
          {rateLimitInfo.reset_time && (
            <span className="text-xs opacity-75">
              Resets: {new Date(rateLimitInfo.reset_time).toLocaleString()}
            </span>
          )}
        </div>
      </motion.div>
    );
  };

  // Check  if input should be disabled
  const isInputDisabled =
    loading || (rateLimitInfo && rateLimitInfo.requests_remaining === 0);
  const placeholderText =
    rateLimitInfo && rateLimitInfo.requests_remaining === 0
      ? "Daily quota exceeded. Try again tomorrow."
      : "Ask me anything...";

  return (
    <>
      {/* Full Screen Chat Overlay */}
      <AnimatePresence>
        <motion.div
          className="fixed inset-0 z-50 bg-white font-inter flex flex-col"
          variants={fullScreenVariants}
          initial="hidden"
          animate="visible"
          exit="exit"
        >
          {/* Header with Rate Limit Status */}
          <motion.div
            className="flex-shrink-0 px-6 py-4 bg-blue-50 border-b border-blue-200"
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
          >
            <div className="max-w-4xl mx-auto">
              <div className="flex justify-between items-center mb-3">
                <span className="text-blue-900 font-semibold text-lg flex items-center gap-3">
                  <motion.img
                    src="https://www.nitj.ac.in/public/assets/images/logo_250.png"
                    alt="Logo"
                    className="w-8 h-8 rounded-full"
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{
                      type: "spring",
                      stiffness: 500,
                      damping: 15,
                      delay: 0.2,
                    }}
                  />
                  <motion.span
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.3 }}
                  >
                    Margdarshak Assistant
                  </motion.span>
                </span>
                <div className="flex items-center gap-3">
                  <motion.button
                    onClick={checkUserRateLimit}
                    className="px-3 py-1 text-sm bg-blue-500 text-white rounded-md hover:bg-blue-600 transition"
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                  >
                    Check Quota
                  </motion.button>
                  <motion.button
                    onClick={toggleChat}
                    className="w-10 h-10 flex items-center justify-center rounded-full text-gray-500 hover:bg-blue-100 hover:text-gray-600 transition text-xl"
                    whileHover={{ scale: 1.1, backgroundColor: "#DBEAFE" }}
                    whileTap={{ scale: 0.9 }}
                  >
                    ✕
                  </motion.button>
                </div>
              </div>

              {/* Rate Limit Status */}
              <RateLimitStatus />

              {/* Rate Limit Error */}
              {rateLimitError && (
                <motion.div
                  className="mt-3 p-3 bg-red-100 border border-red-200 rounded-lg text-red-800 text-sm"
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                >
                  ⚠️ {rateLimitError}
                </motion.div>
              )}

              {/* User IP Display */}
              {userIP && (
                <motion.div
                  className="mt-2 text-xs text-gray-600"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.5 }}
                >
                  Session IP: {userIP}
                </motion.div>
              )}
            </div>
          </motion.div>

          {/* Chat Messages Container */}
          <motion.div
            ref={chatContainerRef}
            className="flex-1 overflow-y-auto px-6 py-8 bg-gray-50"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.2 }}
          >
            <div className="max-w-4xl mx-auto">
              {messages.length === 0 && (
                <motion.div
                  className="text-center text-gray-500 text-lg py-12"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{
                    delay: 0.4,
                    type: "spring",
                    stiffness: 100,
                  }}
                >
                  Hello! I'm your NITJ Margdarshak assistant. How can I help you
                  today?
                </motion.div>
              )}

              <div className="space-y-6">
                {messages.map((msg, index) => (
                  <motion.div
                    key={index}
                    className={`flex w-full ${
                      msg.role === "user" ? "justify-end" : "justify-start"
                    }`}
                    variants={messageVariants}
                    initial="hidden"
                    animate="visible"
                    transition={{
                      delay: 0.1 * index,
                    }}
                  >
                    <div className="max-w-[70%] flex flex-col gap-2">
                      <motion.div
                        className={`text-base shadow-md px-6 py-4 rounded-2xl ${
                          msg.role === "user"
                            ? "bg-blue-500 text-white rounded-br-none"
                            : msg.isError
                            ? "bg-red-50 text-red-800 rounded-bl-none border border-red-200"
                            : "bg-white text-gray-800 rounded-bl-none border border-gray-200"
                        }`}
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{
                          type: "spring",
                          stiffness: 500,
                          damping: 25,
                        }}
                      >
                        {msg.content}
                      </motion.div>

                      {/* Source link below assistant messages */}
                      {msg.role === "assistant" &&
                        msg.source_link_metadata &&
                        !msg.isError && (
                          <motion.div
                            className="text-sm text-blue-600 hover:text-blue-800 underline ml-2"
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            transition={{ delay: 0.3 }}
                          >
                            <a
                              href={msg.source_link_metadata}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="flex items-center gap-2"
                            >
                              <svg
                                xmlns="http://www.w3.org/2000/svg"
                                className="h-4 w-4"
                                fill="none"
                                viewBox="0 0 24 24"
                                stroke="currentColor"
                              >
                                <path
                                  strokeLinecap="round"
                                  strokeLinejoin="round"
                                  strokeWidth={2}
                                  d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1"
                                />
                              </svg>
                              Source: {msg.source_link_metadata}
                            </a>
                          </motion.div>
                        )}
                    </div>
                  </motion.div>
                ))}

                {loading && (
                  <motion.div
                    className="flex justify-start"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: 10 }}
                  >
                    <div className="max-w-[70%] bg-white border border-gray-200 text-gray-800 px-6 py-4 rounded-2xl rounded-bl-none text-base shadow flex items-center gap-3">
                      <span>Thinking</span>
                      <div className="flex gap-1">
                        <motion.div
                          className="w-2 h-2 rounded-full bg-gray-800"
                          variants={typingIndicatorVariants}
                          initial="initial"
                          animate="animate"
                          transition={{ delay: 0 }}
                        />
                        <motion.div
                          className="w-2 h-2 rounded-full bg-gray-800"
                          variants={typingIndicatorVariants}
                          initial="initial"
                          animate="animate"
                          transition={{ delay: 0.2 }}
                        />
                        <motion.div
                          className="w-2 h-2 rounded-full bg-gray-800"
                          variants={typingIndicatorVariants}
                          initial="initial"
                          animate="animate"
                          transition={{ delay: 0.4 }}
                        />
                      </div>
                    </div>
                  </motion.div>
                )}
              </div>
            </div>
          </motion.div>

          {/* Input Section */}
          <motion.div
            className="flex-shrink-0 px-6 py-6 border-t border-gray-200 bg-white"
            variants={inputGroupVariants}
            initial="hidden"
            animate="visible"
          >
            <div className="max-w-4xl mx-auto">
              <motion.div
                className="text-xs text-gray-600 mb-4 text-center"
                initial={{ opacity: 0 }}
                animate={{ opacity: 0.8 }}
                transition={{ delay: 0.4 }}
              >
                <span className="font-semibold text-sm text-red-500">
                  Disclaimer:
                </span>{" "}
                This assistant provides general guidance based on available
                information and is not a substitute for official academic or
                administrative advice. Always consult NIT Jalandhar authorities
                for critical decisions.
              </motion.div>

              <div className="flex gap-4">
                <motion.input
                  type="text"
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder={placeholderText}
                  disabled={isInputDisabled}
                  className={`flex-1 px-4 py-3 border rounded-full text-sm outline-none transition ${
                    isInputDisabled
                      ? "border-gray-300 bg-gray-100 text-gray-500 cursor-not-allowed"
                      : "border-gray-300 focus:border-blue-500 focus:ring-2 focus:ring-blue-200 bg-white"
                  }`}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.3, duration: 0.3 }}
                  whileFocus={
                    !isInputDisabled
                      ? {
                          boxShadow: "0 0 0 3px rgba(59, 130, 246, 0.3)",
                        }
                      : {}
                  }
                />
                <motion.button
                  onClick={sendMessage}
                  disabled={isInputDisabled || !prompt.trim()}
                  className={`px-6 py-3 rounded-full flex items-center justify-center text-base shadow transition ${
                    isInputDisabled || !prompt.trim()
                      ? "bg-gray-300 text-gray-500 cursor-not-allowed"
                      : "bg-blue-500 text-white hover:bg-blue-600"
                  }`}
                  whileHover={
                    !isInputDisabled && prompt.trim()
                      ? {
                          scale: 1.05,
                          backgroundColor: "#2563EB",
                          y: -2,
                          boxShadow: "0 4px 12px rgba(37, 99, 235, 0.3)",
                        }
                      : {}
                  }
                  whileTap={
                    !isInputDisabled && prompt.trim() ? { scale: 0.95 } : {}
                  }
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{
                    delay: 0.4,
                    type: "spring",
                    stiffness: 500,
                    damping: 15,
                  }}
                >
                  {loading ? "..." : "Send"}
                </motion.button>
              </div>
            </div>
          </motion.div>
        </motion.div>
      </AnimatePresence>
    </>
  );
};

export default Chatbot;
