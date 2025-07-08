import React from "react";
import { motion } from "framer-motion";
import MessageBubble from "./MessageBubble";

const ChatMessages = ({ messages, loading, chatContainerRef }) => {
  return (
    <motion.div
      ref={chatContainerRef}
      className="flex-1 overflow-y-auto px-6 py-8 bg-gray-50"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ delay: 0.2 }}
    >
      <div className="max-w-4xl mx-auto space-y-6">
        {messages.length === 0 && (
          <motion.div
            className="text-center text-gray-500 text-lg py-12"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
          >
            Hello! I'm your NITJ Margdarshak assistant. How can I help you today?
          </motion.div>
        )}

        {messages.map((msg, index) => (
          <MessageBubble key={index} msg={msg} index={index} />
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
                {[0, 0.2, 0.4].map((delay, i) => (
                  <motion.div
                    key={i}
                    className="w-2 h-2 rounded-full bg-gray-800"
                    initial={{ scale: 0.8, opacity: 0.5 }}
                    animate={{
                      scale: [0.8, 1, 0.8],
                      opacity: [0.5, 1, 0.5],
                    }}
                    transition={{
                      duration: 1,
                      repeat: Infinity,
                      delay,
                    }}
                  />
                ))}
              </div>
            </div>
          </motion.div>
        )}
      </div>
    </motion.div>
  );
};

export default ChatMessages;
