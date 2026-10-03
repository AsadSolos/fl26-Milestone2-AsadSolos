#include "aiws/chunking_strategy.hpp"
#include "aiws/chunker.hpp"
#include "aiws/context_builder.hpp"
#include "aiws/context_strategy.hpp"
#include "aiws/processing_core.hpp"
#include "aiws/retrieval_engine.hpp"
#include "aiws/retrieval_strategy.hpp"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

int failures = 0;

void check(bool ok, const char* name)
{

    if (!ok)
    {

        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

//one marker chunk per document
class MarkerChunker final : public aiws::ChunkingStrategy
{
public:
    std::vector<aiws::Chunk> chunk(const aiws::Document& d, std::size_t order) const override
    {

        aiws::Chunk c;
        c.id = "marker:" + d.id();
        c.document_id = d.id();
        c.document_order = order;
        c.sequence = 0;
        c.text = "marker";
        c.token_count = 1;
        c.source_begin = 0;
        c.source_end = 0;
        return {c};
    }
};

//always returns a fixed ranking
class ConstantRetrieval final : public aiws::RetrievalStrategy
{
public:
    std::vector<aiws::SearchResult> search(const std::string&, int k, const std::vector<aiws::Chunk>& chunks, const aiws::CorpusIndex&) const override
    {

        std::vector<aiws::SearchResult> out;
        if (k <= 0) return out;
        for (const auto& c : chunks)
        {

            out.push_back(aiws::SearchResult{c.id, c.document_id, c.sequence, c.text, 42.0, 1});
            if (static_cast<int>(out.size()) >= k) break;
        }
        return out;
    }
};

//tags every item as custom
class TaggingContext final : public aiws::ContextStrategy
{
public:
    std::vector<aiws::ContextItem> build(const std::vector<aiws::SearchResult>& ranked, std::size_t budget) const override
    {

        std::vector<aiws::ContextItem> out;
        if (budget == 0) return out;
        for (const auto& r : ranked)
        {

            out.push_back(aiws::ContextItem{r.chunk_id, r.document_id, r.chunk_sequence, "tagged", 1, r.score, false});
        }
        return out;
    }
};

//counts how many times chunk() was called
class CountingChunker final : public aiws::ChunkingStrategy
{
public:
    mutable int calls = 0;

    std::vector<aiws::Chunk> chunk(const aiws::Document& d, std::size_t order) const override
    {

        ++calls;
        aiws::Chunk c;
        c.id = d.id() + "#0";
        c.document_id = d.id();
        c.document_order = order;
        c.sequence = 0;
        c.text = d.text();
        c.token_count = 1;
        return {c};
    }
};

}  // namespace

int main()
{

    using namespace aiws;

    //default config preserves m1 normalization
    check(ProcessingCore::normalize("Hello, WORLD!! 42") == "hello world 42", "default normalize");

    //default config basic search still works
    {

        Workspace ws;
        ws.add_document(Document{"a", "", "alpha beta beta"});
        ws.add_document(Document{"b", "", "gamma"});
        ProcessingCore core;
        core.rebuild(ws);
        check(core.chunk_count() == 2, "default chunks");
        auto r = core.search("beta", 5);
        check(r.size() == 1 && r[0].document_id == "a", "default search");
    }

    //custom chunker observable
    {

        Workspace ws;
        ws.add_document(Document{"a", "", "alpha beta"});
        ProcessingCore core(std::make_unique<MarkerChunker>(), std::make_unique<RetrievalEngine>(), std::make_unique<ContextBuilder>());
        core.rebuild(ws);
        check(core.chunk_count() == 1, "marker count");
        check(core.chunks()[0].id == "marker:a", "marker id");
    }

    //custom retrieval observable
    {

        Workspace ws;
        ws.add_document(Document{"a", "", "alpha beta"});
        ProcessingCore core(std::make_unique<Chunker>(), std::make_unique<ConstantRetrieval>(), std::make_unique<ContextBuilder>());
        core.rebuild(ws);
        auto r = core.search("anything", 1);
        check(r.size() == 1 && r[0].score == 42.0, "constant score");
    }

    //custom context observable
    {

        Workspace ws;
        ws.add_document(Document{"a", "", "alpha beta"});
        ProcessingCore core(std::make_unique<Chunker>(), std::make_unique<RetrievalEngine>(), std::make_unique<TaggingContext>());
        core.rebuild(ws);
        auto c = core.build_context("alpha", 1, 10);
        check(c.size() == 1 && c[0].text == "tagged", "tagged context");
    }

    //null chunking rejected
    {

        bool threw = false;
        try
        {

            ProcessingCore bad(nullptr, std::make_unique<RetrievalEngine>(), std::make_unique<ContextBuilder>());
        }
        catch (const std::invalid_argument&)
        {

            threw = true;
        }
        check(threw, "null chunking rejected");
    }

    //null retrieval rejected
    {

        bool threw = false;
        try
        {

            ProcessingCore bad(std::make_unique<Chunker>(), nullptr, std::make_unique<ContextBuilder>());
        }
        catch (const std::invalid_argument&)
        {

            threw = true;
        }
        check(threw, "null retrieval rejected");
    }

    //null context rejected
    {

        bool threw = false;
        try
        {

            ProcessingCore bad(std::make_unique<Chunker>(), std::make_unique<RetrievalEngine>(), nullptr);
        }
        catch (const std::invalid_argument&)
        {

            threw = true;
        }
        check(threw, "null context rejected");
    }

    //duplicate doc id leaves previous corpus intact
    {

        Workspace good;
        good.add_document(Document{"a", "", "alpha"});
        ProcessingCore core;
        core.rebuild(good);
        const std::size_t before = core.chunk_count();

        Workspace bad;
        bad.add_document(Document{"x", "", "one"});
        bad.add_document(Document{"x", "", "two"});

        bool threw = false;
        try
        {

            core.rebuild(bad);
        }
        catch (const std::invalid_argument&)
        {

            threw = true;
        }
        check(threw, "duplicate rebuild throws");
        check(core.chunk_count() == before, "corpus preserved after failure");
    }

    //move transfers ownership safely
    {

        Workspace ws;
        ws.add_document(Document{"a", "", "alpha"});
        ProcessingCore a(std::make_unique<MarkerChunker>(), std::make_unique<RetrievalEngine>(), std::make_unique<ContextBuilder>());
        a.rebuild(ws);
        ProcessingCore b(std::move(a));
        check(b.chunk_count() == 1 && b.chunks()[0].id == "marker:a", "moved core retains custom chunker");
    }

    //strategies persist across multiple rebuilds
    {

        Workspace ws1;
        ws1.add_document(Document{"a", "", "alpha"});
        auto counter = std::make_unique<CountingChunker>();
        CountingChunker* raw = counter.get();
        ProcessingCore core(std::move(counter), std::make_unique<RetrievalEngine>(), std::make_unique<ContextBuilder>());
        core.rebuild(ws1);
        core.rebuild(ws1);
        check(raw->calls == 2, "strategy reused across rebuilds");
    }

    if (failures) return 1;
    std::cout << "M2 student tests passed\n";
    return 0;
}
