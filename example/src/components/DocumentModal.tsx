import {
  Modal,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';
import { SafeAreaProvider, SafeAreaView } from 'react-native-safe-area-context';
import Ionicons from '@expo/vector-icons/Ionicons';

interface DocumentModalProps {
  visible: boolean;
  onClose: () => void;
  ids: string[];
  document: string;
  onDocumentChange: (text: string) => void;
  onModifyDocument: () => void;
}

export const DocumentModal = ({
  visible,
  onClose,
  ids,
  document,
  onDocumentChange,
  onModifyDocument,
}: DocumentModalProps) => {
  return (
    <Modal visible={visible} animationType="slide" onRequestClose={onClose}>
      <SafeAreaProvider>
        <SafeAreaView style={documentModalStyles.safeArea}>
          <View style={documentModalStyles.header}>
            <Text style={documentModalStyles.title}>Document Editor</Text>
            <TouchableOpacity
              onPress={onClose}
              style={documentModalStyles.closeButton}
            >
              <Ionicons name="close-outline" size={32} color="black" />
            </TouchableOpacity>
          </View>
          <TextInput
            value={document}
            onChangeText={onDocumentChange}
            placeholder="Paste your document here"
            multiline
            style={documentModalStyles.textInput}
          />
          <TouchableOpacity
            onPress={onModifyDocument}
            style={documentModalStyles.addButton}
          >
            <Text style={documentModalStyles.buttonText}>
              {ids.length > 0 ? 'Edit' : 'Add'} Document
            </Text>
          </TouchableOpacity>
        </SafeAreaView>
      </SafeAreaProvider>
    </Modal>
  );
};

const documentModalStyles = StyleSheet.create({
  safeArea: {
    flex: 1,
    padding: 16,
  },
  header: {
    alignItems: 'center',
    marginBottom: 16,
  },
  title: {
    fontSize: 24,
    alignSelf: 'center',
  },
  closeButton: {
    position: 'absolute',
    right: 0,
  },
  textInput: {
    flex: 1,
    borderColor: 'gray',
    backgroundColor: '#e8e8e8',
    borderRadius: 12,
    padding: 12,
    marginBottom: 20,
    textAlignVertical: 'top',
  },
  addButton: {
    backgroundColor: 'black',
    padding: 16,
    borderRadius: 12,
  },
  buttonText: {
    textAlign: 'center',
    color: 'white',
  },
});
