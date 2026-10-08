// Chunked teacher-forced CE; each decoded batch is scored before logits are replaced.
#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdlib>

int main(int argc,char **argv) {
 try {
    if(argc<4||argc>7)throw std::runtime_error("usage: bonsai-eval MODEL INPUT BACKEND_DIR [MAX_NEW=64] [CTX=1024] [BATCH=512]");
    const int max_new=argc>4?std::stoi(argv[4]):64,context=argc>5?std::stoi(argv[5]):1024;
    const int chunk=argc>6?std::stoi(argv[6]):512;
    if(chunk<1||chunk>512||max_new<1||context<max_new+1)throw std::runtime_error("invalid generation/context budget");
    const char *seed_env=std::getenv("VOL2_SAMPLE_SEED");
    const uint32_t seed=seed_env?std::stoul(seed_env):20260927;
    ggml_backend_load_all_from_path(argv[3]);llama_backend_init();
    auto mp=llama_model_default_params();mp.n_gpu_layers=-1;mp.split_mode=LLAMA_SPLIT_MODE_NONE;mp.main_gpu=0;
    auto *model=llama_model_load_from_file(argv[1],mp);
    if(!model)throw std::runtime_error("model load failed");
    auto cp=llama_context_default_params();cp.n_ctx=context;cp.n_batch=chunk;cp.n_ubatch=chunk;cp.n_seq_max=1;
    cp.n_threads=8;cp.n_threads_batch=8;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
    auto *ctx=llama_init_from_model(model,cp);if(!ctx)throw std::runtime_error("context failed");
    auto *vocab=llama_model_get_vocab(model);const int nv=llama_vocab_n_tokens(vocab);
    auto batch=llama_batch_init(chunk,0,1);std::ifstream in(argv[2]);if(!in)throw std::runtime_error("input file missing");
    char mode;long id;int p,n;std::cout<<std::setprecision(12);
    while(in>>mode>>id>>p>>n) {
        if(n<1||n>context||p<1||p>n)throw std::runtime_error("invalid lengths");
        std::vector<llama_token> ids(n);for(auto &t:ids){in>>t;if(t<0||t>=nv)throw std::runtime_error("invalid token");}
        if(mode=='T') {
            std::string hex,raw;in>>hex;
            for(size_t j=0;j<hex.size();j+=2)raw.push_back(char(std::stoi(hex.substr(j,2),nullptr,16)));
            std::vector<llama_token> actual(raw.size()+16);
            int count=llama_tokenize(vocab,raw.data(),raw.size(),actual.data(),actual.size(),false,true);
            if(count!=n||!std::equal(ids.begin(),ids.end(),actual.begin()))throw std::runtime_error("tokenizer mismatch id="+std::to_string(id));
            continue;
        }
        llama_memory_clear(llama_get_memory(ctx),true);
        if(mode=='E'&&p>=n)throw std::runtime_error("CE requires nonempty target");
        if(mode=='G'&&n+max_new>context)throw std::runtime_error("generation exceeds declared context budget");
        const auto started=std::chrono::steady_clock::now();
        const int input_count=mode=='E'?n-1:n;
        double nll=0.;int scored_tokens=0;
        for(int start=0;start<input_count;start+=chunk) {
            batch.n_tokens=std::min(chunk,input_count-start);
            for(int i=0;i<batch.n_tokens;i++) {
                batch.token[i]=ids[start+i];batch.pos[i]=start+i;batch.n_seq_id[i]=1;batch.seq_id[i][0]=0;
                batch.logits[i]=mode=='E'?start+i>=p-1:start+i==input_count-1;
            }
            if(llama_decode(ctx,batch)!=0)throw std::runtime_error("decode failed");
            if(mode=='E') {
                llama_synchronize(ctx);
                const int first=std::max(0,p-1-start);
                std::vector<const float*> logits;
                for(int i=first;i<batch.n_tokens;i++) {
                    auto *l=llama_get_logits_ith(ctx,i);
                    if(!l)throw std::runtime_error("missing batch logits");
                    logits.push_back(l);
                }
                double batch_nll=0.;
                #pragma omp parallel for num_threads(8) reduction(+:batch_nll)
                for(int row=0;row<int(logits.size());row++) {
                    const float *l=logits[row];double max=*std::max_element(l,l+nv),sum=0.;
                    for(int j=0;j<nv;j++)sum+=std::exp(double(l[j])-max);
                    batch_nll+=std::log(sum)+max-l[ids[start+first+row+1]];
                }
                nll+=batch_nll;scored_tokens+=int(logits.size());
            }
        }
        if(mode=='E') {
            if(scored_tokens!=n-p)throw std::runtime_error("target token count mismatch");
            if(!std::isfinite(nll))throw std::runtime_error("nonfinite NLL");
            std::cout<<"{\"kind\":\"ce\",\"id\":"<<id<<",\"tokens\":"<<n-p<<",\"nll\":"<<nll<<"}"<<std::endl;
        } else if(mode=='L') {
            const float *l=llama_get_logits_ith(ctx,-1);
            std::cout<<"{\"kind\":\"logits\",\"id\":"<<id<<",\"logits\":[";
            for(int j=0;j<nv;j++){if(j)std::cout<<",";std::cout<<l[j];}
            std::cout<<"]}"<<std::endl;
        } else if(mode=='G') {
            const double prefill_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
            auto *sampler=llama_sampler_chain_init(llama_sampler_chain_default_params());
            llama_sampler_chain_add(sampler,llama_sampler_init_top_k(20));
            llama_sampler_chain_add(sampler,llama_sampler_init_top_p(0.95f,1));
            llama_sampler_chain_add(sampler,llama_sampler_init_temp(1.0f));
            llama_sampler_chain_add(sampler,llama_sampler_init_dist(seed+uint32_t(id)));
            std::cerr<<"sampling id="<<id<<" temp=1 top_k=20 top_p=.95 seed="<<(seed+uint32_t(id))<<std::endl;
            std::vector<int> generated;bool ended=false;
            for(int step=0;step<max_new;step++) {
                const float *l=llama_get_logits_ith(ctx,-1);if(!l)throw std::runtime_error("missing generation logits");
                int token=llama_sampler_sample(sampler,ctx,-1);
                if(llama_vocab_is_eog(vocab,token)){ended=true;break;}
                generated.push_back(token);if(step==max_new-1)break;batch.n_tokens=1;batch.token[0]=token;batch.pos[0]=n+step;
                batch.n_seq_id[0]=1;batch.seq_id[0][0]=0;batch.logits[0]=1;
                if(llama_decode(ctx,batch)!=0)throw std::runtime_error("generation failed");
            }
            llama_sampler_free(sampler);
            std::cout<<"{\"kind\":\"generation\",\"id\":"<<id<<",\"tokens\":[";
            for(size_t j=0;j<generated.size();j++){if(j)std::cout<<",";std::cout<<generated[j];}
            std::cout<<"],\"seconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()<<",\"prefill_seconds\":"<<prefill_seconds<<",\"finish_reason\":\""<<(ended?"eos":"length")<<"\"}"<<std::endl;
        } else throw std::runtime_error("unknown mode");
    }
    llama_batch_free(batch);llama_free(ctx);llama_model_free(model);llama_backend_free();
    return 0;
 } catch(const std::exception &e) {std::cerr<<e.what()<<std::endl;return 1;}
}
