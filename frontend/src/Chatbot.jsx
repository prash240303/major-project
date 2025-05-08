import React, { useState, useEffect, useRef } from "react";
import axios from "axios";

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:8000";

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

      setMessages([
        ...newMessages,
        { role: "assistant", content: data.answer || "No response available." },
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

  return (
    <div className="fixed bottom-6 right-6 z-50 font-inter">
      {!chatExpanded && (
        <button
          onClick={toggleChat}
          className="w-fit h-fit p-0 bg-white overflow-hidden cursor-pointer rounded-full border border-blue-500 shadow-lg hover:scale-105 hover:shadow-xl transition-transform duration-300"
        >
          <img src="/icon.webp" alt="Chat Icon" className="w-[72px] h-[72px]" />
        </button>
      )}

      {chatExpanded && (
        <div className="w-[400px] max-h-[85vh] bg-white rounded-xl shadow-xl flex flex-col overflow-hidden border border-gray-200 animate-slide-in font-inter">
          <div className="flex justify-between items-center px-5 py-4 bg-blue-100 border-b border-blue-200">
            <span className="text-blue-900 font-semibold text-sm flex items-center gap-2">
              <img
                src="https://www.nitj.ac.in/public/assets/images/logo_250.png"
                alt="Logo"
                className="w-6 h-6 rounded-full"
              />
              Margdarshak Assistant
            </span>
            <button
              onClick={toggleChat}
              className="w-7 h-7 flex items-center justify-center rounded-full text-gray-500 hover:bg-blue-50 hover:text-gray-600 transition"
            >
              ✕
            </button>
          </div>

          <div
            ref={chatContainerRef}
            className="flex flex-col gap-4 px-5 py-8 bg-gray-50 flex-grow overflow-y-auto"
          >
            {messages.length === 0 && (
              <div className="text-center text-gray-500 text-base">
                Hello! I'm your NITJ Margdarshak assistant. How can I help you
                today?
              </div>
            )}

            {messages.map((msg, index) => (
              <div
                key={index}
                className={`flex flex-col gap-1 ${
                  msg.role === "user" ? "justify-end" : "justify-start"
                } mb-2`}
              >
                <div
                  className={`max-w-[85%] text-sm shadow-md px-4 py-3 rounded-2xl ${
                    msg.role === "user"
                      ? "bg-blue-500 text-white rounded-br-none"
                      : "bg-blue-100 text-gray-800 rounded-bl-none"
                  }`}
                >
                  {msg.content}
                </div>

                {/* Disclaimer for assistant messages only */}
                {msg.role !== "user" && <></>}
              </div>
            ))}

            {loading && (
              <div className="mr-auto max-w-[85%] mb-4 bg-blue-200 text-gray-800 px-4 py-3 rounded-2xl rounded-bl-none text-sm shadow flex items-center gap-2">
                <span>Thinking</span>
                <div className="flex gap-1">
                  <div className="w-2 h-2 rounded-full bg-gray-800 animate-bounce delay-[-0.32s]" />
                  <div className="w-2 h-2 rounded-full bg-gray-800 animate-bounce delay-[-0.16s]" />
                  <div className="w-2 h-2 rounded-full bg-gray-800 animate-bounce" />
                </div>
              </div>
            )}
          </div>

          <div className="flex flex-col gap-3 items-start px-5 pb-4 pt-1 border-t border-gray-200 bg-white">
            <div className="text-xs text-gray-500 italic ml-2 mt-1 max-w-[85%]">
              Disclaimer: This assistant provides general guidance based on
              available information and is not a substitute for official
              academic or administrative advice. Always consult NITJ authorities
              for critical decisions.
            </div>
            <div className="flex w-full">
              <input
                type="text"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask me anything..."
                className="flex-grow px-4 py-3 border border-gray-300 rounded-full text-sm outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-200 transition"
              />
              <button
                onClick={sendMessage}
                className="ml-2 p-2 bg-blue-500 text-white rounded-full w-10 h-10 flex items-center justify-center text-sm hover:bg-blue-600 hover:-translate-y-[1px] shadow transition-transform"
              >
                ➤
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Chatbot;
