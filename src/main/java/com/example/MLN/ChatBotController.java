package com.example.MLN;

import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.media.Content;
import lombok.RequiredArgsConstructor;
import org.springframework.ai.chat.memory.ChatMemory;
import org.springframework.ai.chat.messages.Message;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.List;

@RestController
@RequestMapping("/api/chat")
@RequiredArgsConstructor
public class ChatBotController {
    private final ChatBotService chatBotService;
    private final ChatMemory chatMemory;

    @PostMapping
    public ResponseEntity<ChatBotResponse> chat(
            @RequestBody ChatRequest request
    ) {
        ChatBotResponse response = ChatBotResponse.builder()
                .answer(chatBotService.chat(request.getQuestion(), request.getConversationId()))
                .build();
        return ResponseEntity.ok(response);
    }

    @PostMapping(value = "/rag", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<ChatBotResponse> rag(
            @Parameter(content = @Content(mediaType = MediaType.MULTIPART_FORM_DATA_VALUE))
            @RequestParam("file") MultipartFile file
    ) {
        chatBotService.rag(file);

        return ResponseEntity.ok(
                ChatBotResponse.builder()
                        .answer("RAG process completed")
                        .build()
        );
    }

    @GetMapping("/history/{conversationId}")
    public List<Message> getHistory(@PathVariable String conversationId) {
        return chatMemory.get(conversationId);
    }
}