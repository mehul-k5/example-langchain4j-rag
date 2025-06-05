package org.example;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.parser.TextDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.bgesmallenv15q.BgeSmallEnV15QuantizedEmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.util.List;

import static dev.langchain4j.data.document.loader.FileSystemDocumentLoader.loadDocument;
import static dev.langchain4j.internal.Utils.getOrDefault;
import static org.example.Utils.*;

public class ExampleRAG {

    /**
     * This example demonstrates how to implement a naive Retrieval-Augmented Generation (RAG) application.
     * By "naive", we mean that we won't use any advanced RAG techniques.
     * In each interaction with the Large Language Model (LLM), we will:
     * 1. Take the user's query as-is.
     * 2. Embed it using an embedding model.
     * 3. Use the query's embedding to search an embedding store (containing small segments of your documents)
     * for the X most relevant segments.
     * 4. Append the found segments to the user's query.
     * 5. Send the combined input (user query + segments) to the LLM.
     * 6. Hope that:
     * - The user's query is well-formulated and contains all necessary details for retrieval.
     * - The found segments are relevant to the user's query.
     */
    private static final String API_KEY =  getOrDefault(System.getenv("API_KEY"), "demo");

    public static void main(String[] args) {

        // Let's create an assistant that will know about our document
        AiAssistant assistant = createAssistant("documents1/policy.txt");

        startConversationWith(assistant);
    }

    private static AiAssistant createAssistant(String documentPath) {

        //create a chat model, also known as a LLM, which will answer our queries.
        ChatModel chatModel = GoogleAiGeminiChatModel.builder()
                .apiKey(API_KEY)
                .modelName("gemini-2.0-flash")
                .build();

        // load a document
        DocumentParser documentParser = new TextDocumentParser();
        Document document = loadDocument(toPath(documentPath), documentParser);


        // Now, we need to split this document into smaller segments, also known as "chunks."
        // This approach allows us to send only relevant segments to the LLM in response to a user query,
        // rather than the entire document. For instance, if a user asks about room rent policies,
        // we will identify and send only those segments related to room rent.
        // A good starting point is to use a recursive document splitter that initially attempts
        // to split by paragraphs. If a paragraph is too large to fit into a single segment,
        // the splitter will recursively divide it by newlines, then by sentences, and finally by words,
        // if necessary, to ensure each piece of text fits into a single segment.
        DocumentSplitter splitter = DocumentSplitters.recursive(300, 0);
        List<TextSegment> segments = splitter.split(document);

//        System.out.println(segments.toString());

        // Now, we need to embed (also known as "vectorize") these segments.
        // Embedding is needed for performing similarity searches.
        // For this example, we'll use a local in-process embedding model.
        //Langchain4j supports various ONNX (Open Neural Network Exchange) models
        EmbeddingModel embeddingModel = new BgeSmallEnV15QuantizedEmbeddingModel();
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();

//        System.out.println(embeddings);

        // Next, we will store these embeddings in an embedding store (also known as a "vector database").
        // This store will be used to search for relevant segments during each interaction with the LLM.
        // For simplicity, this example uses an in-memory embedding store, but you can choose from any supported store.
        // Langchain4j currently supports more than 15 popular embedding stores.
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);

        // The content retriever is responsible for retrieving relevant content based on a user query.

        ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2) // on each interaction we will retrieve the 2 most relevant segments
                .minScore(0.5) // we want to retrieve segments at least somewhat similar to user query
                .build();


        // Optionally, we can use a chat memory, enabling back-and-forth conversation with the LLM
        // and allowing it to remember previous interactions.
        ChatMemory chatMemory = MessageWindowChatMemory.withMaxMessages(10);


        // The final step is to build our AI Service,
        // configuring it to use the components we've created above.
        return AiServices.builder(AiAssistant.class)
                .chatModel(chatModel)
                .contentRetriever(contentRetriever)
                .chatMemory(chatMemory)
                .build();
    }
}
