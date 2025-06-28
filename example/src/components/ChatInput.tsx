import { TextInput, TouchableOpacity, View, StyleSheet } from 'react-native';
import Ionicons from '@expo/vector-icons/Ionicons';

interface ChatInputProps {
  message: string;
  onMessageChange: (text: string) => void;
  onAddDocument: () => void;
  onToggleAugmentedGeneration: () => void;
  onMessageSubmit: () => void;
  augmentedGeneration: boolean;
  isReady: boolean;
  isGenerating: boolean;
}

export const ChatInput = ({
  message,
  onMessageChange,
  onAddDocument,
  onToggleAugmentedGeneration,
  onMessageSubmit,
  isReady,
  isGenerating,
  augmentedGeneration,
}: ChatInputProps) => {
  const messageSubmitBtnDisabled = !isReady || isGenerating || !message.trim();

  return (
    <View style={chatInputStyles.container}>
      <TextInput
        value={message}
        onChangeText={onMessageChange}
        multiline
        placeholder="Ask anything"
        placeholderTextColor="gray"
      />
      <View style={chatInputStyles.actionsContainer}>
        <TouchableOpacity onPress={onAddDocument}>
          <Ionicons name="document-text-outline" size={28} color="black" />
        </TouchableOpacity>
        <View style={chatInputStyles.rightActions}>
          <TouchableOpacity
            onPress={onToggleAugmentedGeneration}
            style={chatInputStyles.toggleButton}
          >
            <Ionicons name="search-outline" size={28} />
            {!augmentedGeneration && (
              <View style={chatInputStyles.disabledOverlay} />
            )}
          </TouchableOpacity>
          <TouchableOpacity
            onPress={onMessageSubmit}
            disabled={messageSubmitBtnDisabled}
            style={
              messageSubmitBtnDisabled && chatInputStyles.submitButtonDisabled
            }
          >
            <Ionicons name="arrow-up-circle" size={36} color="black" />
          </TouchableOpacity>
        </View>
      </View>
    </View>
  );
};

const chatInputStyles = StyleSheet.create({
  container: {
    backgroundColor: '#e8e8e8',
    borderRadius: 12,
    padding: 12,
    gap: 8,
    margin: 12,
  },
  actionsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  rightActions: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  toggleButton: {
    position: 'relative',
  },
  disabledOverlay: {
    backgroundColor: 'black',
    width: 28,
    height: 2,
    borderRadius: 1,
    position: 'absolute',
    left: 'auto',
    top: 14,
    transform: [{ rotate: '-50deg' }],
  },
  submitButtonDisabled: {
    opacity: 0.5,
  },
});
