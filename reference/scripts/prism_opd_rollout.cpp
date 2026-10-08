// Full-support T=1 sampling, exact chosen-token log probabilities including EOS.
// No conversion of generated token pieces to UTF8: byte tokens remain aligned.
#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
struct Job { long id; unsigned seed; std::vector<llama_token> prompt, out; std::vector<double> lp; llama_sampler *sampler=nullptr; bool ended=false; };
int main(int argc, char **argv) {
 try {
  if(argc!=7) throw std::runtime_error("MODEL INPUT BACKEND MAX_NEW CTX_PER_SLOT SLOTS");
  int budget=std::stoi(argv[4]), context=std::stoi(argv[5]), slots=std::stoi(argv[6]);
  if(budget<1 || slots<1 || slots>32 || context<budget+1) throw std::runtime_error("invalid limits");
  std::ifstream input(argv[2]); std::vector<Job> jobs; long id; unsigned seed; int n;
  while(input>>id>>seed>>n) {
   if(n<1 || n+budget>context) throw std::runtime_error("invalid prompt length");
   Job j; j.id=id; j.seed=seed; j.prompt.resize(n);
   for(auto &t:j.prompt) if(!(input>>t)) throw std::runtime_error("incomplete input");
   jobs.push_back(std::move(j));
  }
  if(jobs.empty()) throw std::runtime_error("no jobs");
  ggml_backend_load_all_from_path(argv[3]); llama_backend_init();
  auto mp=llama_model_default_params(); mp.n_gpu_layers=-1; mp.split_mode=LLAMA_SPLIT_MODE_NONE; mp.main_gpu=0;
  auto *model=llama_model_load_from_file(argv[1],mp); if(!model) throw std::runtime_error("model load failed");
  auto cp=llama_context_default_params(); cp.n_ctx=slots*context; cp.n_seq_max=slots;
  cp.n_batch=512; cp.n_ubatch=512; cp.n_threads=8; cp.n_threads_batch=8; cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
  auto *ctx=llama_init_from_model(model,cp); if(!ctx) throw std::runtime_error("context failed");
  auto *vocab=llama_model_get_vocab(model); int nv=llama_vocab_n_tokens(vocab);
  auto batch=llama_batch_init(512,0,1); std::cout<<std::setprecision(12);
  auto sample=[&](Job &j, int row) {
   const float *l=llama_get_logits_ith(ctx,row); if(!l) throw std::runtime_error("missing logits");
   double m=*std::max_element(l,l+nv), sum=0.;
   #pragma omp parallel for num_threads(4) reduction(+:sum)
   for(int k=0;k<nv;k++) sum+=std::exp(double(l[k])-m);
   auto t=llama_sampler_sample(j.sampler,ctx,row);
   double logp=double(l[t])-m-std::log(sum);
   if(!std::isfinite(logp) || logp>1e-6) throw std::runtime_error("nonfinite probability");
   j.out.push_back(t); j.lp.push_back(logp); j.ended=llama_vocab_is_eog(vocab,t);
  };
  for(size_t base=0;base<jobs.size();base+=slots) {
   auto started=std::chrono::steady_clock::now(); int count=std::min(size_t(slots),jobs.size()-base);
   llama_memory_clear(llama_get_memory(ctx),true);
   for(int s=0;s<count;s++) {
    auto &j=jobs[base+s]; j.sampler=llama_sampler_init_dist(j.seed);
    for(int start=0;start<int(j.prompt.size());start+=512) {
     batch.n_tokens=std::min(512,int(j.prompt.size())-start);
     for(int i=0;i<batch.n_tokens;i++) {
      auto t=j.prompt[start+i]; if(t<0 || t>=nv) throw std::runtime_error("invalid token");
      batch.token[i]=t; batch.pos[i]=start+i; batch.n_seq_id[i]=1; batch.seq_id[i][0]=s;
      batch.logits[i]=(start+i==int(j.prompt.size())-1);
     }
     if(llama_decode(ctx,batch)) throw std::runtime_error("prefill failed");
    }
    sample(j,-1);
   }
   for(int step=1;step<budget;step++) {
    std::vector<int> active; batch.n_tokens=0;
    for(int s=0;s<count;s++) if(!jobs[base+s].ended) {
     auto &j=jobs[base+s]; int i=batch.n_tokens++;
     batch.token[i]=j.out.back(); batch.pos[i]=j.prompt.size()+j.out.size()-1;
     batch.n_seq_id[i]=1; batch.seq_id[i][0]=s; batch.logits[i]=true; active.push_back(s);
    }
    if(active.empty()) break;
    if(llama_decode(ctx,batch)) throw std::runtime_error("generation failed");
    for(int i=0;i<int(active.size());i++) sample(jobs[base+active[i]],i);
    if(step%128==0) std::cerr<<"generated="<<step+1<<" active="<<active.size()<<std::endl;
   }
   double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
   for(int s=0;s<count;s++) {
    auto &j=jobs[base+s]; std::cout<<"{\"id\":"<<j.id<<",\"tokens\":[";
    for(size_t k=0;k<j.out.size();k++){if(k)std::cout<<",";std::cout<<j.out[k];}
    std::cout<<"],\"behavior_logp\":[";
    for(size_t k=0;k<j.lp.size();k++){if(k)std::cout<<",";std::cout<<j.lp[k];}
    std::cout<<"],\"finish_reason\":\""<<(j.ended?"eos":"length")<<"\",\"group_seconds\":"<<seconds<<"}"<<std::endl;
    llama_sampler_free(j.sampler);
   }
  }
  llama_batch_free(batch); llama_free(ctx); llama_model_free(model); llama_backend_free(); return 0;
 } catch(const std::exception &e) { std::cerr<<e.what()<<std::endl; return 1; }
}
