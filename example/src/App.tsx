import { type Message, useRAG } from 'react-native-rag';
import { OPSQLiteVectorStore } from '@react-native-rag/op-sqlite';
import {
  QWEN3_0_6B_QUANTIZED,
  ALL_MINILM_L6_V2,
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
  const [downloadProgress, setDownloadProgress] = useState(0);
  const [modalVisible, setModalVisible] = useState(false);
  const [document, setDocument] = useState<string>('');
  const [ids, setIds] = useState<string[]>([]);
  const [augmentedGeneration, setAugmentedGeneration] = useState(true);
  const [message, setMessage] = useState<string>('');
  const [messages, setMessages] = useState<Message[]>([]);

  const vectorStore = useMemo(() => {
    return new OPSQLiteVectorStore({
      name: 'rag_example_db1',
      embeddings: new ExecuTorchEmbeddings(ALL_MINILM_L6_V2),
    });
  }, []);

  const llm = useMemo(() => {
    return new ExecuTorchLLM({
      ...QWEN3_0_6B_QUANTIZED,
      onDownloadProgress: setDownloadProgress,
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
          await rag.deleteDocument({ predicate: (value) => value.id === id });
        }
        setIds([]);
      }
      const newIds = await rag.splitAddDocument({ document });
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
      const result = await rag.generate({
        input: [...messages, newMessage],
        augmentedGeneration,
      });
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
          ) : !rag.isReady ? (
            <View style={appStyles.loadingContainer}>
              <Text style={appStyles.loadingText}>
                Loading {(downloadProgress * 100).toFixed(2)}%
              </Text>
            </View>
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
            isReady={rag.isReady}
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
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    fontSize: 20,
    fontWeight: 'bold',
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
