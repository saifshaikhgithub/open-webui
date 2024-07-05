<script lang="ts">
    import { onMount } from 'svelte';
    import { docgen } from '$lib/apis/docgen';
    import { user, WEBUI_NAME } from '$lib/stores';
    import Navbar from '$lib/components/layout/Navbar.svelte';
    import MessageInput from '$lib/components/chat/MessageInput.svelte';
    import Message from '$lib/components/chat/Message.svelte';

    let basePrompt = '';
    let stepPrompt = '';
    let messages = [];
    let loading = false;
    let loadingPercentage = 0;
    let isSubmitting = false;
    let abortController = null;

    function updateProgress(startAt, endAt, increment = 5, interval = 100) {
        loadingPercentage = startAt;
        const intervalId = setInterval(() => {
            if (loadingPercentage >= endAt) {
                clearInterval(intervalId);
                if (endAt === 100 && !abortController.signal.aborted) {
                    loading = false;
                }
            } else {
                loadingPercentage += increment;
            }
        }, interval);
    }

    const performDocGeneration = async () => {
        if (!basePrompt || !stepPrompt) {
	    console.log(basePrompt);
	    console.log(stepPrompt);
            console.error("Please enter prompts for document generation");
            return;
        }
	console.log(basePrompt);
	console.log(stepPrompt);
        abortController = new AbortController();
        loading = true;
        isSubmitting = true;
        updateProgress(0, 50);

        try {
            const response = await docgen(basePrompt, stepPrompt);
	    console.log("test");
            if (abortController.signal.aborted) {
                throw new Error("AbortError");
            }

            updateProgress(50, 100, 10, 50);

            const newMessages = [
                ...messages,
                { id: Date.now(), content: basePrompt, role: 'user'},
                { id: Date.now() + 1, content: response.document, role: 'assistant', label: 'Generated Document:'},
            ];

            messages = newMessages;
            basePrompt = '';
            stepPrompt = '';
        } catch (error) {
            if (error.name === 'AbortError' || error.message === 'AbortError') {
                console.log("Document generation request was canceled.");
            } else {
                console.error("Failed to perform document generation:", error);
            }
        } finally {
            loading = false;
            loadingPercentage = 0;
            isSubmitting = false;
            abortController = null;
        }
    };

    function cancelDocGeneration() {
        if (abortController) {
            abortController.abort();
            messages = [...messages, { id: Date.now(), content: "Document generation request canceled by the user.", role: 'system' }];
            loading = false;
            loadingPercentage = 0;
            isSubmitting = false;
        }
    }

</script>

<svelte:head>
    <title>Document Generation</title>
</svelte:head>

<div class="h-screen max-h-[100dvh] w-full flex flex-col">
    <Navbar />
    <input bind:value={basePrompt} placeholder="Enter base"/>
    <MessageInput bind:prompt={stepPrompt} submitPrompt={performDocGeneration}/>
    {#if isSubmitting}
        <button on:click={cancelDocGeneration}>Cancel Generation</button>
    {/if}
    <div class="h-full w-full flex flex-col py-8">
        <Message {messages} />
    </div>
    {#if loading}
        <div class="loading-indicator">
            Hello... {Math.round(loadingPercentage)}%
        </div>
    {/if}
</div>

<style>
    .loading-indicator {
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-size: 24px;
        color: white;
        background-color: rgba(0, 0, 0, 0.7);
        padding: 10px 20px;
        border-radius: 10px;
    }
    button {
        position: fixed;
        bottom: 20px;
        right: 20px;
        padding: 10px 20px;
        font-size: 16px;
        color: white;
        background-color: blue;
        border: none;
        border-radius: 5px;
        cursor: pointer;
    }
    button:hover {
        background-color: #0056b3;
    }
</style>
