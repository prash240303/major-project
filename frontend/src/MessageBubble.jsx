import React from "react";
import { motion } from "framer-motion";

const MessageBubble = ({ msg, index }) => {
  const isUser = msg.role === "user";

  return (
    <motion.div
      className={`flex w-full ${isUser ? "justify-end" : "justify-start"}`}
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.1 * index }}
    >
      <div className="max-w-[70%] flex flex-col gap-2">
        <motion.div
          className={`text-base shadow-md px-6 py-4 rounded-2xl ${
            isUser
              ? "bg-blue-500 text-white rounded-br-none"
              : msg.isError
              ? "bg-red-50 text-red-800 rounded-bl-none border border-red-200"
              : "bg-white text-gray-800 rounded-bl-none border border-gray-200"
          }`}
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ type: "spring", stiffness: 500, damping: 25 }}
        >
          {msg.content}
        </motion.div>

        {msg.role === "assistant" && msg.source_link_metadata && !msg.isError && (
          <motion.div
            className="text-sm text-blue-600 hover:text-blue-800 underline ml-2"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.3 }}
          >
            <a href={msg.source_link_metadata} target="_blank" rel="noopener noreferrer">
              🔗 Source: {msg.source_link_metadata}
            </a>
          </motion.div>
        )}
      </div>
    </motion.div>
  );
};

export default MessageBubble;
