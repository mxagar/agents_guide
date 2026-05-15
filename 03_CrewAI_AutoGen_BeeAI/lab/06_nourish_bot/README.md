## NourishBot Exercise

This folder contains the completed standalone notebook for the NourishBot exercise in `Instructions.pdf`.
The original project is included unchanged as the `NourishBot` Git submodule, pointing to the `5-final` branch of `HaileyTQuach/Smart-Nutritional-App`.

The original NourishBot repo is a Gradio-based AI nutrition coach. Users upload a food image, optionally provide dietary restrictions, and choose between two workflows:

* **Recipe workflow:** detect ingredients from the image, filter them against dietary restrictions, and generate recipe suggestions.
* **Analysis workflow:** analyze a meal image and return estimated calories, nutrient breakdown, health evaluation, and a disclaimer.

The notebook [`06_nourish_bot.ipynb`](./06_nourish_bot.ipynb) recreates those core agentic workflows without Gradio. It uses current CrewAI style, OpenAI models, `python-dotenv`, Tavily search for recipe context, custom `@tool` functions, and Pydantic structured outputs.

### What Happens in the Notebook

* It first checks that the original `NourishBot` submodule files are present. The submodule is reference material only; the notebook defines its own standalone workflow.
* It loads environment variables with `load_dotenv()` and initializes:
  * `LLM(model="openai/gpt-4o")` for CrewAI agents
  * `OpenAI()` for direct multimodal image calls
  * `TavilySearchTool()` for current recipe context
* It defines Pydantic schemas for structured outputs:
  * `Recipe` and `RecipeSuggestionOutput` for the recipe workflow
  * `NutrientAnalysisOutput` and nested nutrient models for the analysis workflow
* It defines an image helper that converts a local image file into a base64 data URL so it can be sent to an OpenAI vision-capable model.
* It defines three custom CrewAI tools with `@tool`:
  * `extract_ingredients`: identifies ingredients from a food image
  * `clean_ingredients`: normalizes raw ingredient text into a list
  * `analyze_food_image`: produces nutrition and health guidance from a food image
* It creates three agents:
  * `ingredient_agent`: extracts and cleans ingredients
  * `nutrition_agent`: analyzes calories, nutrients, and health balance
  * `recipe_agent`: suggests recipes and can use Tavily search
* It builds two crews:
  * `recipe_crew`: runs ingredient detection, then recipe generation with `context=[ingredient_detection_task]`
  * `analysis_crew`: runs direct meal-image nutrition analysis
* The final cells show optional `kickoff(...)` calls. They are commented out so the notebook can be reviewed without calling OpenAI or Tavily.

Required `.env` values:

```bash
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
```

Main files:

* `NourishBot/`: unchanged original repository submodule.
* `Instructions.pdf`: original exercise instructions.
* `06_nourish_bot.ipynb`: standalone notebook implementing the core recipe and nutrition analysis workflows.
