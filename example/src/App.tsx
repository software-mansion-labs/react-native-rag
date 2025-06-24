import { type Message, MemoryVectorStore, useRAG } from 'react-native-rag';
import {
  LLAMA3_2_1B_QLORA,
  LLAMA3_2_1B_TOKENIZER,
  LLAMA3_2_TOKENIZER_CONFIG,
  ALL_MINILM_L6_V2,
  ALL_MINILM_L6_V2_TOKENIZER,
} from 'react-native-executorch';
import { useMemo, useState } from 'react';
import {
  ExecuTorchEmbeddings,
  ExecuTorchLLM,
} from '@react-native-rag/executorch';
import {
  KeyboardAvoidingView,
  Text,
  View,
  StyleSheet,
  Platform,
  Alert,
} from 'react-native';
import { SafeAreaProvider, SafeAreaView } from 'react-native-safe-area-context';
import SWMIcon from '../assets/icons/swm_icon.svg';
import { MessagesList } from './components/MessagesList';
import { ChatInput } from './components/ChatInput';
import { DocumentModal } from './components/DocumentModal';

export default function App() {
  const [modalVisible, setModalVisible] = useState(false);
  const [document, setDocument] = useState<string>('');
  const [ids, setIds] = useState<string[]>([]);
  const [augmentedGeneration, setAugmentedGeneration] = useState(true);
  const [message, setMessage] = useState<string>('');
  const [messages, setMessages] = useState<Message[]>([]);

  const vectorStore = useMemo(() => {
    return new MemoryVectorStore({
      embeddings: new ExecuTorchEmbeddings({
        modelSource: ALL_MINILM_L6_V2,
        tokenizerSource: ALL_MINILM_L6_V2_TOKENIZER,
      }),
    });
  }, []);

  const llm = useMemo(() => {
    return new ExecuTorchLLM({
      modelSource: LLAMA3_2_1B_QLORA,
      tokenizerSource: LLAMA3_2_1B_TOKENIZER,
      tokenizerConfigSource: LLAMA3_2_TOKENIZER_CONFIG,
    });
  }, []);

  const rag = useRAG({ vectorStore, llm });

  const modifyDocument = async () => {
    if (!rag.isReady) {
      console.error('RAG is not initialized');
      Alert.alert('Error', 'RAG system is not ready. Please try again.');
      return;
    }
    try {
      if (ids.length) {
        for (const id of ids) {
          await rag.deleteDocument(id);
        }
        setIds([]);
      }
      const newIds = await rag.splitAddDocument(document);
      setIds(newIds);
      console.log('Document splitted and added with IDs:', newIds);
      setModalVisible(false);
    } catch (error) {
      console.error('Error adding/modifying document:', error);
      Alert.alert('Error', 'Failed to add/modify document. Please try again.');
    }
  };

  const handleMessageSubmit = async () => {
    if (!rag.isReady) {
      console.warn('RAG not ready');
      return;
    }
    if (!message.trim()) {
      console.warn('Message is empty');
      return;
    }

    const newMessage: Message = {
      role: 'user',
      content: message,
    };

    setMessages((prevMessages) => [...prevMessages, newMessage]);
    setMessage('');

    try {
      const result = await rag.generate(
        [...messages, newMessage],
        augmentedGeneration
      );
      setMessages((prevMessages) => [
        ...prevMessages,
        { role: 'assistant', content: result },
      ]);
    } catch (error) {
      console.error('Error generating response:', error);
      Alert.alert('Error', 'Failed to generate response. Please try again.');
    }
  };

  const openDocumentModal = () => {
    setModalVisible(true);
  };

  const handleAugmentedGeneration = () => {
    setAugmentedGeneration((prev) => !prev);
  };

  return (
    <SafeAreaProvider>
      <SafeAreaView style={appStyles.safeArea}>
        <SWMIcon width={45} height={45} style={appStyles.headerIcon} />
        <KeyboardAvoidingView
          style={appStyles.keyboardAvoidingView}
          behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        >
          {messages.length > 0 ? (
            <MessagesList
              messages={messages}
              response={rag.response}
              isGenerating={rag.isGenerating}
            />
          ) : (
            <View style={appStyles.emptyStateContainer}>
              <Text style={appStyles.emptyStateTitle}>Hello! 👋</Text>
              <Text style={appStyles.emptyStateSubtitle}>
                What can I help you with?
              </Text>
            </View>
          )}
          <ChatInput
            message={message}
            onMessageChange={setMessage}
            onAddDocument={openDocumentModal}
            onToggleAugmentedGeneration={handleAugmentedGeneration}
            onMessageSubmit={handleMessageSubmit}
            augmentedGeneration={augmentedGeneration}
            isGenerating={rag.isGenerating}
          />
        </KeyboardAvoidingView>
      </SafeAreaView>
      <DocumentModal
        visible={modalVisible}
        onClose={() => setModalVisible(false)}
        ids={ids}
        document={document}
        onDocumentChange={setDocument}
        onModifyDocument={modifyDocument}
      />
    </SafeAreaProvider>
  );
}

const appStyles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: '#FFFFFF',
  },
  headerIcon: {
    alignSelf: 'center',
    marginVertical: 8,
  },
  keyboardAvoidingView: {
    flex: 1,
  },
  emptyStateContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 20,
  },
  emptyStateTitle: {
    fontSize: 30,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  emptyStateSubtitle: {
    fontSize: 20,
    textAlign: 'center',
  },
});
