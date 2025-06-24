import type { Message } from 'react-native-rag';
import { View, ScrollView, Text, StyleSheet } from 'react-native';
import LlamaIcon from '../../assets/icons/llama_icon.svg';
import { useRef } from 'react';

interface MessagesListProps {
  messages: Message[];
  response: string;
  isGenerating: boolean;
}

export const MessagesList = ({
  messages,
  response,
  isGenerating,
}: MessagesListProps) => {
  const scrollRef = useRef<ScrollView>(null);

  return (
    <ScrollView
      ref={scrollRef}
      onContentSizeChange={() => {
        scrollRef.current?.scrollToEnd({ animated: false });
      }}
      style={messagesListStyles.container}
    >
      {messages.map((message, index) => (
        <View
          key={index}
          style={
            message.role === 'assistant'
              ? messagesListStyles.assistantMessageContainer
              : messagesListStyles.userMessageContainer
          }
        >
          {message.role === 'assistant' && (
            <View style={messagesListStyles.iconContainer}>
              <LlamaIcon width={24} height={24} />
            </View>
          )}
          <Text
            style={[
              messagesListStyles.messageText,
              message.role === 'user' && messagesListStyles.userMessageText,
            ]}
          >
            {message.content}
          </Text>
        </View>
      ))}
      {isGenerating && (
        <View style={messagesListStyles.assistantMessageContainer}>
          <View style={messagesListStyles.iconContainer}>
            <LlamaIcon width={24} height={24} />
          </View>
          {!response ? (
            <View>
              <Text>Generating...</Text>
            </View>
          ) : (
            <Text style={messagesListStyles.responseText}>
              {response.trim()}
            </Text>
          )}
        </View>
      )}
    </ScrollView>
  );
};

const messagesListStyles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 8,
  },
  assistantMessageContainer: {
    flexDirection: 'row',
    marginVertical: 8,
  },
  userMessageContainer: {
    flexDirection: 'row-reverse',
    marginVertical: 8,
    paddingHorizontal: 12,
    paddingVertical: 8,
    maxWidth: '75%',
    borderRadius: 12,
    alignSelf: 'flex-end',
    alignItems: 'center',
    backgroundColor: 'black',
  },
  iconContainer: {
    borderRadius: 12,
    height: 32,
    width: 32,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#e8e8e8',
    marginRight: 8,
  },
  messageText: {
    fontSize: 14,
    lineHeight: 19.6,
    flexShrink: 1,
  },
  userMessageText: {
    color: 'white',
  },
  responseText: {
    fontSize: 14,
    lineHeight: 19.6,
    flexShrink: 1,
  },
});
