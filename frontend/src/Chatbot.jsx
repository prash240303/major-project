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

  // Add Inter font to the document if it's not already added
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

  const styles = {
    fontStyle: {
      fontFamily: "'Inter', sans-serif",
    },
    container: {
      position: "fixed",
      bottom: "24px",
      right: "24px",
      zIndex: 50,
    },
    chatButton: {
      height: "60px",
      width: "60px",
      backgroundColor: "#3B82F6",
      color: "white",
      borderRadius: "9999px",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      boxShadow:
        "0 10px 25px -5px rgba(59, 130, 246, 0.5), 0 8px 10px -6px rgba(59, 130, 246, 0.3)",
      fontSize: "24px",
      transition: "all 0.3s ease",
      cursor: "pointer",
      border: "none",
      fontFamily: "'Inter', sans-serif",
    },
    chatButtonHover: {
      transform: "scale(1.05)",
      boxShadow:
        "0 20px 25px -5px rgba(59, 130, 246, 0.4), 0 10px 10px -5px rgba(59, 130, 246, 0.2)",
    },
    chatWindow: {
      width: "400px",
      maxHeight: "85vh",
      backgroundColor: "white",
      borderRadius: "16px",
      boxShadow:
        "0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)",
      display: "flex",
      flexDirection: "column",
      overflow: "hidden",
      border: "1px solid rgba(229, 231, 235, 0.8)",
      animation: "slideIn 0.3s ease-out",
      fontFamily: "'Inter', sans-serif",
    },
    header: {
      display: "flex",
      justifyContent: "space-between",
      alignItems: "center",
      padding: "16px 20px",
      backgroundColor: "#EFF6FF",
      borderBottom: "1px solid #DBEAFE",
      fontFamily: "'Inter', sans-serif",
    },
    headerTitle: {
      fontWeight: 600,
      fontSize: "16px",
      color: "#1E40AF",
      display: "flex",
      alignItems: "center",
      gap: "8px",
      fontFamily: "'Inter', sans-serif",
    },
    logoIcon: {
      fontSize: "20px",
    },
    closeButton: {
      color: "#6B7280",
      background: "none",
      border: "none",
      cursor: "pointer",
      borderRadius: "50%",
      width: "28px",
      height: "28px",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      transition: "all 0.2s",
      fontFamily: "'Inter', sans-serif",
    },
    closeButtonHover: {
      backgroundColor: "rgba(238, 242, 255, 0.8)",
      color: "#4B5563",
    },
    messageContainer: {
      flexGrow: 1,
      overflowY: "auto",
      padding: "20px",
      gap: "16px",
      backgroundColor: "#F9FAFB",
      display: "flex",
      flexDirection: "column",
      fontFamily: "'Inter', sans-serif",
    },
    userMessage: {
      padding: "12px 16px",
      borderRadius: "18px 18px 0 18px",
      backgroundColor: "#3B82F6",
      color: "white",
      textAlign: "left",
      marginLeft: "auto",
      maxWidth: "85%",
      marginBottom: "16px",
      boxShadow: "0 2px 5px rgba(59, 130, 246, 0.2)",
      fontFamily: "'Inter', sans-serif",
      fontWeight: 400,
      lineHeight: 1.5,
      fontSize: "14px",
    },
    assistantMessage: {
      padding: "12px 16px",
      borderRadius: "18px 18px 18px 0",
      backgroundColor: "#BFD7FF",
      color: "#1F2937",
      marginRight: "auto",
      maxWidth: "85%",
      marginBottom: "16px",
      boxShadow: "0 2px 5px rgba(0, 0, 0, 0.05)",
      fontFamily: "'Inter', sans-serif",
      fontWeight: 400,
      lineHeight: 1.5,
      fontSize: "14px",
    },
    loadingIndicator: {
      padding: "12px 16px",
      borderRadius: "18px 18px 18px 0",
      backgroundColor: "#BFD7FF",
      color: "#1F2937",
      marginRight: "auto",
      maxWidth: "85%",
      marginBottom: "16px",
      boxShadow: "0 2px 5px rgba(0, 0, 0, 0.05)",
      fontFamily: "'Inter', sans-serif",
      fontWeight: 400,
      lineHeight: 1.5,
      fontSize: "14px",
      display: "flex",
      alignItems: "center",
      gap: "8px",
      fontFamily: "'Inter', sans-serif",
    },
    loadingDots: {
      display: "flex",
      gap: "4px",
    },
    dot: {
      width: "8px",
      height: "8px",
      borderRadius: "50%",
      backgroundColor: "#1F2937",
      animation: "bounce 1.4s infinite ease-in-out both",
    },
    inputContainer: {
      display: "flex",
      alignItems: "center",
      padding: "16px 20px",
      borderTop: "1px solid #E5E7EB",
      backgroundColor: "white",
      fontFamily: "'Inter', sans-serif",
    },
    input: {
      flexGrow: 1,
      padding: "12px 16px",
      border: "1px solid #E5E7EB",
      borderRadius: "9999px",
      fontSize: "14px",
      outline: "none",
      transition: "border-color 0.2s, box-shadow 0.2s",
      fontFamily: "'Inter', sans-serif",
    },
    inputFocus: {
      borderColor: "#3B82F6",
      boxShadow: "0 0 0 2px rgba(59, 130, 246, 0.2)",
    },
    sendButton: {
      marginLeft: "8px",
      padding: "10px",
      backgroundColor: "#3B82F6",
      color: "white",
      borderRadius: "9999px",
      fontSize: "14px",
      border: "none",
      cursor: "pointer",
      width: "40px",
      height: "40px",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      transition: "all 0.2s",
      boxShadow: "0 2px 5px rgba(59, 130, 246, 0.3)",
      fontFamily: "'Inter', sans-serif",
    },
    sendButtonHover: {
      backgroundColor: "#2563EB",
      transform: "translateY(-1px)",
      boxShadow: "0 4px 6px rgba(59, 130, 246, 0.3)",
    },
    messageTime: {
      fontSize: "10px",
      color: "#9CA3AF",
      marginTop: "4px",
      textAlign: "right",
      fontFamily: "'Inter', sans-serif",
    },
    welcomeMessage: {
      textAlign: "center",
      color: "#6B7280",
      padding: "16px",
      fontSize: "14px",
      fontFamily: "'Inter', sans-serif",
    },
  };

  return (
    <div style={styles.container}>
      {!chatExpanded && (
        <button
          onClick={toggleChat}
          style={styles.chatButton}
          onMouseOver={(e) => {
            e.currentTarget.style.transform = "scale(1.05)";
            e.currentTarget.style.boxShadow =
              "0 20px 25px -5px rgba(59, 130, 246, 0.4), 0 10px 10px -5px rgba(59, 130, 246, 0.2)";
          }}
          onMouseOut={(e) => {
            e.currentTarget.style.transform = "scale(1)";
            e.currentTarget.style.boxShadow =
              "0 10px 25px -5px rgba(59, 130, 246, 0.5), 0 8px 10px -6px rgba(59, 130, 246, 0.3)";
          }}
        >
          <img
            src="/robot.png"
            alt=""
            style={{
              width: "36px",
              height: "36px",
            }}
          />
        </button>
      )}

      {chatExpanded && (
        <div style={styles.chatWindow}>
          <div style={styles.header}>
            <span style={styles.headerTitle}>
              <span style={styles.logoIcon}>
                <img
                  src="https://www.nitj.ac.in/public/assets/images/logo_250.png"
                  alt="Logo"
                  style={{
                    width: "24px",
                    height: "24px",
                    borderRadius: "50%",
                    marginRight: "8px",
                  }}
                />
              </span>{" "}
              Margdarshak Assistant
            </span>
            <button
              onClick={toggleChat}
              style={styles.closeButton}
              onMouseOver={(e) => {
                e.currentTarget.style.backgroundColor =
                  "rgba(238, 242, 255, 0.8)";
                e.currentTarget.style.color = "#4B5563";
              }}
              onMouseOut={(e) => {
                e.currentTarget.style.backgroundColor = "transparent";
                e.currentTarget.style.color = "#6B7280";
              }}
            >
              ✕
            </button>
          </div>

          <div ref={chatContainerRef} style={styles.messageContainer}>
            {messages.length === 0 && (
              <div style={styles.welcomeMessage}>
                Hello! I'm your NITJ Margdarshak assistant. How can I help you
                today?
              </div>
            )}

            {messages.map((msg, index) => (
              <div key={index}>
                <div
                  style={
                    msg.role === "user"
                      ? styles.userMessage
                      : styles.assistantMessage
                  }
                >
                  {msg.content}
                </div>
              </div>
            ))}

            {loading && (
              <div style={styles.loadingIndicator}>
                <span>Thinking</span>
                <div style={styles.loadingDots}>
                  <div
                    style={{ ...styles.dot, animationDelay: "-0.32s" }}
                  ></div>
                  <div
                    style={{ ...styles.dot, animationDelay: "-0.16s" }}
                  ></div>
                  <div style={{ ...styles.dot, animationDelay: "0s" }}></div>
                </div>
              </div>
            )}
          </div>

          <div style={styles.inputContainer}>
            <input
              type="text"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask me anything..."
              style={styles.input}
              onFocus={(e) => {
                e.currentTarget.style.borderColor = "#3B82F6";
                e.currentTarget.style.boxShadow =
                  "0 0 0 2px rgba(59, 130, 246, 0.2)";
              }}
              onBlur={(e) => {
                e.currentTarget.style.borderColor = "#E5E7EB";
                e.currentTarget.style.boxShadow = "none";
              }}
            />
            <button
              onClick={sendMessage}
              style={styles.sendButton}
              onMouseOver={(e) => {
                e.currentTarget.style.backgroundColor = "#2563EB";
                e.currentTarget.style.transform = "translateY(-1px)";
                e.currentTarget.style.boxShadow =
                  "0 4px 6px rgba(59, 130, 246, 0.3)";
              }}
              onMouseOut={(e) => {
                e.currentTarget.style.backgroundColor = "#3B82F6";
                e.currentTarget.style.transform = "translateY(0)";
                e.currentTarget.style.boxShadow =
                  "0 2px 5px rgba(59, 130, 246, 0.3)";
              }}
            >
              ➤
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default Chatbot;
