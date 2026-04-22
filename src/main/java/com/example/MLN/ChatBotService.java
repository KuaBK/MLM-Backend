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

    public String chat(String question, String userId) {
        return chatClient.prompt()
                .options(GoogleGenAiChatOptions.builder()
                        .model("gemini-3.1-flash-lite-preview")
                        .build())
                .advisors(a -> a.param(ChatMemory.CONVERSATION_ID, userId))
                .advisors(QuestionAnswerAdvisor.builder(vectorStore).build())
                .user(u -> u.text(question))
                .call()
                .content();
    }

    public void rag(List<MultipartFile> files) {
        List<Resource> resourceList = convertResources(files);
        ingestResources(resourceList);
    }

    private List<Resource> convertResources(List<MultipartFile> files) {
        if (files == null || files.isEmpty()) return List.of();
        List<Resource> result = new ArrayList<>();
        for (MultipartFile file : files) {
            if (file.isEmpty()) continue;

            try {
                Resource resource = new ByteArrayResource(file.getBytes()) {
                    @Override
                    public String getFilename() {
                        return file.getOriginalFilename();
                    }
                };
                result.add(resource);
            } catch (Exception e) {
                throw new InternalException("Failed to read file: " + file.getOriginalFilename(), e);
            }
        }

        return result;
    }

    private void ingestResources(List<Resource> resources) {
        try {
            TextSplitter splitter = new TokenTextSplitter();

            List<CompletableFuture<List<Document>>> futures = resources.stream()
                    .map(resource -> CompletableFuture.supplyAsync(() -> {
                        try {
                            TikaDocumentReader reader = new TikaDocumentReader(resource);
                            List<Document> docs = splitter.split(reader.read());
                            docs.forEach(d -> d.getMetadata().put("filename", resource.getFilename()));
                            return docs;
                        } catch (Exception e) {
                            log.error("[ChatBot] Lỗi khi đọc file {}: {}", resource.getFilename(), e.getMessage(), e);
                            return List.<Document>of();
                        }
                    }))
                    .toList();

            List<Document> allDocuments = CompletableFuture.allOf(futures.toArray(new CompletableFuture[0]))
                    .thenApply(v -> futures.stream()
                            .flatMap(f -> f.join().stream())
                            .toList())
                    .join();

            if (!allDocuments.isEmpty()) {
                vectorStore.accept(allDocuments);
                log.info("[ChatBot] Đã ingest {} document chunks từ {} file(s)",
                        allDocuments.size(), resources.size());
            }
        } catch (Exception ex) {
            log.error("[ChatBot] Lỗi khi ingest file: {}", ex.getMessage(), ex);
        }
    }
}
