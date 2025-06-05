package org.example;

/**
 * This is an "AI Service". It is a Java service with AI capabilities/features.
 * The goal is to seamlessly integrate AI functionality into your (existing) codebase with minimal friction.
 * LangChain4j then provides an implementation for this interface using proxy and reflection.
 */
public interface AiAssistant {

    String answer(String query);
}
