import React, { useState, useEffect, useRef } from "react";
import axios from "axios";
import { motion, AnimatePresence } from "framer-motion";

// const BACKEND_URL = "https://margdarshak-backend.onrender.com"|| ""; // Uncomment this line for production
// const BACKEND_URL = "http://15.207.109.149:8000";
const BACKEND_URL = "https://margdarshak.tech";

const Chatbot = () => {
  const [chatExpanded, setChatExpanded] = useState(false);
  const [messages, setMessages] = useState([]);
  const [conversationId, setConversationId] = useState(null);
  const [prompt, setPrompt] = useState("");
  const [loading, setLoading] = useState(false);
  const chatContainerRef = useRef(null);

  const toggleChat = () => setChatExpanded((prev) => !prev);

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
    return () => {
      document.head.removeChild(link);
    };
  }, []);

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
    <div className="fixed bottom-6 right-6 z-50 font-inter">
      <AnimatePresence>
        {!chatExpanded && (
          <motion.button
            onClick={toggleChat}
            className="w-fit h-fit p-0 bg-white overflow-hidden cursor-pointer rounded-full border border-blue-500 shadow-lg hover:scale-105 hover:shadow-xl transition-transform duration-300"
            initial={{ opacity: 0, scale: 0.5 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.5 }}
            whileHover={buttonVariants.hover}
            whileTap={buttonVariants.tap}
            transition={{ type: "spring", stiffness: 500, damping: 15 }}
          >
            <img
              src="/icon.webp"
              alt="Chat Icon"
              className="w-[72px] h-[72px]"
            />
          </motion.button>
        )}

        {chatExpanded && (
          <motion.div
            className="w-[400px] max-h-[85vh] bg-white rounded-xl shadow-xl flex flex-col overflow-hidden border border-gray-200 font-inter"
            variants={chatWindowVariants}
            initial="hidden"
            animate="visible"
            exit="exit"
          >
            <motion.div
              className="flex justify-between items-center px-5 py-4 bg-blue-100 border-b border-blue-200"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.1 }}
            >
              <span className="text-blue-900 font-semibold text-sm flex items-center gap-2">
                <motion.img
                  src="https://www.nitj.ac.in/public/assets/images/logo_250.png"
                  alt="Logo"
                  className="w-6 h-6 rounded-full"
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
                className="w-7 h-7 flex items-center justify-center rounded-full text-gray-500 hover:bg-blue-50 hover:text-gray-600 transition"
                whileHover={{ scale: 1.1, backgroundColor: "#EBF5FF" }}
                whileTap={{ scale: 0.9 }}
              >
                ✕
              </motion.button>
            </motion.div>

            <motion.div
              ref={chatContainerRef}
              className="flex flex-col gap-4 px-5 py-8 bg-gray-50 flex-grow overflow-y-auto"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.2 }}
            >
              {messages.length === 0 && (
                <motion.div
                  className="text-center text-gray-500 text-base"
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

              {messages.map((msg, index) => (
                <motion.div
                  key={index}
                  className={`flex w-full flex-col gap-1 ${
                    msg.role === "user"
                      ? "justify-end items-end"
                      : "justify-start"
                  } mb-2`}
                  variants={messageVariants}
                  initial="hidden"
                  animate="visible"
                  transition={{
                    delay: 0.1 * index,
                  }}
                >
                  <motion.div
                    className={`max-w-[95%] text-sm shadow-md px-4 py-3 rounded-2xl ${
                      msg.role === "user"
                        ? "bg-blue-500 text-white rounded-br-none"
                        : "bg-blue-100 text-gray-800 rounded-bl-none"
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
                      className="text-xs text-blue-600 truncate max-w-xs hover:text-blue-800 underline ml-2 mt-1"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ delay: 0.3 }}
                    >
                      <a
                        href={msg.source_link_metadata}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex items-center gap-1"
                      >
                        <svg
                          xmlns="http://www.w3.org/2000/svg"
                          className="h-3 w-3"
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
                        Source : {msg.source_link_metadata}
                      </a>
                    </motion.div>
                  )}
                </motion.div>
              ))}

              {loading && (
                <motion.div
                  className="mr-auto max-w-[85%] mb-4 bg-blue-200 text-gray-800 px-4 py-3 rounded-2xl rounded-bl-none text-sm shadow flex items-center gap-2"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: 10 }}
                >
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
                </motion.div>
              )}
            </motion.div>

            <motion.div
              className="flex flex-col gap-3 items-start px-5 pb-4 pt-1 border-t border-gray-200 bg-white"
              variants={inputGroupVariants}
              initial="hidden"
              animate="visible"
            >
              <motion.div
                className="text-xs text-gray-500  ml-2 mt-1 max-w-[85%]"
                initial={{ opacity: 0 }}
                animate={{ opacity: 0.8 }}
                transition={{ delay: 0.4 }}
              >
                <span className="font-semibold text-red-500">Disclaimer:</span>
                This assistant provides general guidance based on available
                information and is not a substitute for official academic or
                administrative advice. Always consult NITJ authorities for
                critical decisions.
              </motion.div>
              <div className="flex w-full">
                <motion.input
                  type="text"
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Ask me anything..."
                  className="flex-grow px-4 py-3 border border-gray-300 rounded-full text-sm outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-200 transition"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.3, duration: 0.3 }}
                  whileFocus={{
                    boxShadow: "0 0 0 3px rgba(59, 130, 246, 0.3)",
                  }}
                />
                <motion.button
                  onClick={sendMessage}
                  className="ml-2 p-2 bg-blue-500 text-white rounded-full w-10 h-10 flex items-center justify-center text-sm shadow"
                  whileHover={{
                    scale: 1.05,
                    backgroundColor: "#2563EB",
                    y: -2,
                    boxShadow: "0 4px 12px rgba(37, 99, 235, 0.3)",
                  }}
                  whileTap={{ scale: 0.95 }}
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{
                    delay: 0.4,
                    type: "spring",
                    stiffness: 500,
                    damping: 15,
                  }}
                >
                  ➤
                </motion.button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default Chatbot;
