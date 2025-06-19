import React, { useState, useEffect, useRef } from "react";
import axios from "axios";
import { motion, AnimatePresence } from "framer-motion";

// const BACKEND_URL = "https://margdarshak-backend.onrender.com"|| "";
// const BACKEND_URL = "http://15.207.109.149:8000";
// const BACKEND_URL = "http://localhost:8000"; // For local development
const BACKEND_URL = "https://margdarshak.tech";

const Chatbot = () => {
  const [chatExpanded, setChatExpanded] = useState(false);
  const [messages, setMessages] = useState([]);
  const [conversationId, setConversationId] = useState(null);
  const [prompt, setPrompt] = useState("");
  const [loading, setLoading] = useState(false);
  const chatContainerRef = useRef(null);

  const toggleChat = () => {
    setChatExpanded((prev) => !prev);
    
    // Handle body overflow when chat is expanded/collapsed
    if (!chatExpanded) {
      // Expanding - hide body overflow
      document.body.style.overflow = 'hidden';
    } else {
      // Collapsing - restore body overflow
      document.body.style.overflow = '';
    }
  };

  const sendMessage = async () => {
    if (!prompt.trim()) return;

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
      console.log("payload", payload);
      const response = await axios.post(`${BACKEND_URL}/chat`, payload);
      const data = response.data;
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
    } catch (error) {
      setMessages([
        ...newMessages,
        { role: "assistant", content: `Error: ${error.message}` },
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

    document.addEventListener('keydown', handleGlobalKeyDown);
    
    return () => {
      if (document.head.contains(link)) {
        document.head.removeChild(link);
      }
      document.removeEventListener('keydown', handleGlobalKeyDown);
      // Cleanup: restore body overflow when component unmounts
      document.body.style.overflow = '';
    };
  }, [chatExpanded]);

  // Animation variants
  const chatWindowVariants = {
    hidden: { opacity: 0, scale: 0.8, y: 20 },
    visible: {
      opacity: 1,
      scale: 1,
      y: 0,
      transition: {
        type: "spring",
        damping: 25,
        stiffness: 300,
      },
    },
    exit: {
      opacity: 0,
      scale: 0.8,
      y: 20,
      transition: {
        duration: 0.2,
      },
    },
  };

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

  const buttonVariants = {
    hover: {
      scale: 1.05,
      boxShadow: "0px 5px 15px rgba(0, 0, 0, 0.1)",
      transition: {
        type: "spring",
        stiffness: 400,
        damping: 10,
      },
    },
    tap: {
      scale: 0.95,
      transition: {
        type: "spring",
        stiffness: 400,
        damping: 10,
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

  return (
    <>
      {/* Chat Button - Only show when not expanded */}
      <AnimatePresence>
        {!chatExpanded && (
          <motion.div
            className="fixed bottom-6 right-6 z-50 font-inter"
            initial={{ opacity: 0, scale: 0.5 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.5 }}
            transition={{ type: "spring", stiffness: 500, damping: 15 }}
          >
            <motion.button
              onClick={toggleChat}
              className="w-24 h-24 p-0 bg-white overflow-hidden cursor-pointer rounded-full border border-blue-500 shadow-lg hover:scale-105 hover:shadow-xl transition-transform duration-300"
              whileHover={buttonVariants.hover}
              whileTap={buttonVariants.tap}
            >
              <img
                src="/icon.webp"
                alt="Chat Icon"
                className="w-[72px] h-[72px]"
              />
            </motion.button>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Full Screen Chat Overlay */}
      <AnimatePresence>
        {chatExpanded && (
          <motion.div
            className="fixed inset-0 z-50 bg-white font-inter flex flex-col"
            variants={fullScreenVariants}
            initial="hidden"
            animate="visible"
            exit="exit"
          >
            {/* Header */}
            <motion.div
              className="flex justify-between items-center px-6 py-4 bg-blue-100 border-b border-blue-200 flex-shrink-0"
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
            >
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
              <motion.button
                onClick={toggleChat}
                className="w-10 h-10 flex items-center justify-center rounded-full text-gray-500 hover:bg-blue-50 hover:text-gray-600 transition text-xl"
                whileHover={{ scale: 1.1, backgroundColor: "#EBF5FF" }}
                whileTap={{ scale: 0.9 }}
              >
                ✕
              </motion.button>
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
                    Hello! I'm your NITJ Margdarshak assistant. How can I help you today?
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
                        {msg.role === "assistant" && msg.source_link_metadata && (
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
                  className="text-sm text-gray-600 mb-4 text-center"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 0.8 }}
                  transition={{ delay: 0.4 }}
                >
                  <span className="font-semibold text-red-500">Disclaimer:</span>
                  {" "}This assistant provides general guidance based on available
                  information and is not a substitute for official academic or
                  administrative advice. Always consult NITJ authorities for
                  critical decisions.
                </motion.div>
                
                <div className="flex gap-4">
                  <motion.input
                    type="text"
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder="Ask me anything..."
                    className="flex-1 px-6 py-4 border border-gray-300 rounded-full text-base outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-200 transition"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.3, duration: 0.3 }}
                    whileFocus={{
                      boxShadow: "0 0 0 3px rgba(59, 130, 246, 0.3)",
                    }}
                  />
                  <motion.button
                    onClick={sendMessage}
                    disabled={!prompt.trim() || loading}
                    className="px-8 py-4 bg-blue-500 text-white rounded-full flex items-center justify-center text-base shadow disabled:opacity-50 disabled:cursor-not-allowed"
                    whileHover={!loading && prompt.trim() ? {
                      scale: 1.05,
                      backgroundColor: "#2563EB",
                      y: -2,
                      boxShadow: "0 4px 12px rgba(37, 99, 235, 0.3)",
                    } : {}}
                    whileTap={!loading && prompt.trim() ? { scale: 0.95 } : {}}
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
        )}
      </AnimatePresence>
    </>
  );
};

export default Chatbot;