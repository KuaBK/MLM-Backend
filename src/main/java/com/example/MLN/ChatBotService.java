package com.example.MLN;

import lombok.extern.slf4j.Slf4j;
import org.apache.logging.log4j.util.InternalException;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.vectorstore.QuestionAnswerAdvisor;
import org.springframework.ai.chat.memory.ChatMemory;
import org.springframework.ai.document.Document;
import org.springframework.ai.google.genai.GoogleGenAiChatOptions;
import org.springframework.ai.reader.tika.TikaDocumentReader;
import org.springframework.ai.transformer.splitter.TextSplitter;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.core.io.Resource;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;

@Slf4j
@Service
public class ChatBotService {
    private final ChatClient chatClient;
    private final VectorStore vectorStore;

    public ChatBotService(
            ChatClient chatClient,
            VectorStore vectorStore) {
        this.chatClient = chatClient;
        this.vectorStore = vectorStore;
    }

    public String chat(String question, String conversationId) {
        return chatClient.prompt()
                .options(GoogleGenAiChatOptions.builder()
                        .model("gemini-3.1-flash-lite-preview")
                        .build())
                .advisors(a -> a.param(ChatMemory.CONVERSATION_ID, conversationId))
                .advisors(QuestionAnswerAdvisor.builder(vectorStore).build())
                .user(u -> u.text(question))
                .call()
                .content();
    }

    public void rag(MultipartFile file) {
        if (file == null || file.isEmpty()) {
            throw new InternalException("File is empty");
        }

        Resource resource = convertResource(file);
        ingestResource(resource);
    }

    private Resource convertResource(MultipartFile file) {
        try {
            return new ByteArrayResource(file.getBytes()) {
                @Override
                public String getFilename() {
                    return file.getOriginalFilename();
                }
            };
        } catch (Exception e) {
            throw new InternalException("Failed to read file: " + file.getOriginalFilename(), e);
        }
    }

    private void ingestResource(Resource resource) {
        try {
            TextSplitter splitter = new TokenTextSplitter();

            TikaDocumentReader reader = new TikaDocumentReader(resource);
            log.info("File name: {}", resource.getFilename());

            List<Document> docs = splitter.split(reader.read());

            log.info("Docs size: {}", docs.size());

            docs.forEach(d -> d.getMetadata().put("filename", resource.getFilename()));

            if (!docs.isEmpty()) {
                vectorStore.accept(docs);
                log.info("[ChatBot] Đã ingest {} document chunks từ file {}",
                        docs.size(), resource.getFilename());
            }

        } catch (Exception e) {
            log.error("[ChatBot] Lỗi khi ingest file {}: {}",
                    resource.getFilename(), e.getMessage(), e);
        }
    }
}
