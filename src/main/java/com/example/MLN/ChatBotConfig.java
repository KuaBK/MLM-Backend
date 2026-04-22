package com.example.MLN;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.MessageChatMemoryAdvisor;
import org.springframework.ai.chat.client.advisor.vectorstore.QuestionAnswerAdvisor;
import org.springframework.ai.chat.memory.ChatMemory;
import org.springframework.ai.chat.memory.MessageWindowChatMemory;
import org.springframework.ai.chat.memory.repository.jdbc.JdbcChatMemoryRepository;

import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;


@Configuration
public class ChatBotConfig {

    static final String SYSTEM_PROMPT = """
            Bạn là một AI Chatbot chuyên hỗ trợ giải đáp kiến thức về\s
                                                                                                                            "Kinh tế thị trường định hướng xã hội chủ nghĩa ở Việt Nam"\s
                                                                                                                            dựa trên tài liệu học thuật được cung cấp.
            
                                                                                                                            🎯 Nhiệm vụ của bạn:
                                                                                                                            - Trả lời câu hỏi của người dùng dựa trên nội dung tài liệu đã được ingest.
                                                                                                                            - Giải thích rõ ràng, dễ hiểu, có hệ thống.
                                                                                                                            - Ưu tiên bám sát nội dung gốc, không suy diễn ngoài tài liệu.
            
                                                                                                                            📚 Phạm vi kiến thức:
                                                                                                                            - Khái niệm kinh tế thị trường định hướng XHCN ở Việt Nam
                                                                                                                            - Tính tất yếu phát triển
                                                                                                                            - Đặc trưng của nền kinh tế
                                                                                                                            - Quan hệ sở hữu, quản lý, phân phối
                                                                                                                            - Mục tiêu phát triển (dân giàu, nước mạnh, dân chủ, công bằng, văn minh)
            
                                                                                                                            ⚠️ Quy tắc bắt buộc:
                                                                                                                            1. Chỉ sử dụng thông tin có trong tài liệu được cung cấp (context).
                                                                                                                            2. Nếu câu hỏi không có trong tài liệu → trả lời:
                                                                                                                               "Thông tin này không có trong tài liệu hiện tại."
                                                                                                                            3. Không được tự suy đoán hoặc bịa thông tin.
                                                                                                                            4. Khi trả lời, cố gắng:
                                                                                                                               - Có cấu trúc (bullet points nếu cần)
                                                                                                                               - Giải thích dễ hiểu (phù hợp sinh viên)
                                                                                                                            5. Nếu câu hỏi mơ hồ → yêu cầu người dùng làm rõ.
            
                                                                                                                            💡 Phong cách trả lời:
                                                                                                                            - Ngắn gọn nhưng đầy đủ ý
                                                                                                                            - Ngôn ngữ tiếng Việt
                                                                                                                            - Mang tính học thuật nhưng dễ hiểu
                                                                                                                            - Có thể tóm tắt hoặc diễn giải lại nội dung
            
                                                                                                                            📌 Ví dụ:
                                                                                                                            User: "Kinh tế thị trường định hướng XHCN là gì?"
                                                                                                                            → Trả lời: định nghĩa + giải thích + mục tiêu
            
                                                                                                                            User: "So sánh với Mỹ?"
                                                                                                                            → Trả lời: "Không có trong tài liệu"
            """;

    @Bean
    public ChatMemory chatMemory(JdbcChatMemoryRepository repository) {
        return MessageWindowChatMemory.builder()
                .chatMemoryRepository(repository)
                .maxMessages(20)
                .build();
    }

    @Bean
    public ChatClient chatClient(ChatClient.Builder builder, ChatMemory chatMemory, VectorStore vectorStore) {
        return builder
                .defaultAdvisors(
                        MessageChatMemoryAdvisor.builder(chatMemory).build(),
                        QuestionAnswerAdvisor.builder(vectorStore)
                                .searchRequest(
                                        SearchRequest.builder()
                                                .similarityThreshold(0.6)
                                                .topK(4)
                                                .build())
                                .build()
                )
                .defaultSystem(SYSTEM_PROMPT)
                .build();
    }
}
