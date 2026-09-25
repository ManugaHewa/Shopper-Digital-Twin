"""Hand-written design of the fake store: departments, categories and shopper segments.

Nothing in this file is random. The generator turns these tables into thousands
of products and shoppers, adding randomness around the values set here.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Category:
    department: str
    name: str
    staple: bool  # bought on a repeat cycle (milk) rather than occasionally (a TV)
    cycle_days: float  # typical days between purchases for a staple; 0 for discretionary
    median_price: float  # typical shelf price in dollars
    breadth: float = 1.0  # relative number of products stocked in the category
    peak_day: int = 0  # day of year when demand peaks (only used if season_amp > 0)
    season_amp: float = 0.0  # 0 = no seasonality, 1 = demand swings from 0x to 2x
    requires: str | None = None  # "dog", "cat", "baby" or "car": only those shoppers buy it


def S(dept, name, cycle, price, breadth=1.0, peak=0, amp=0.0, requires=None):
    """Staple category shorthand."""
    return Category(dept, name, True, cycle, price, breadth, peak, amp, requires)


def D(dept, name, price, breadth=1.0, peak=0, amp=0.0, requires=None):
    """Discretionary category shorthand."""
    return Category(dept, name, False, 0.0, price, breadth, peak, amp, requires)


CATEGORIES: list[Category] = [
    # Grocery
    S("Grocery", "Milk", 7, 3.5),
    S("Grocery", "Eggs", 10, 3.0, 0.6),
    S("Grocery", "Cheese", 14, 4.5),
    S("Grocery", "Yogurt", 7, 4.0, 0.9),
    S("Grocery", "Fresh Fruit", 6, 4.0),
    S("Grocery", "Fresh Vegetables", 6, 3.5),
    S("Grocery", "Bread", 6, 2.8, 0.8),
    S("Grocery", "Pasta & Rice", 21, 2.5, 0.9),
    S("Grocery", "Canned Goods", 21, 1.8),
    S("Grocery", "Breakfast Cereal", 14, 4.2, 1.2),
    S("Grocery", "Coffee", 21, 8.5, 1.2),
    S("Grocery", "Tea", 30, 5.0, 0.7),
    S("Grocery", "Condiments & Sauces", 30, 3.5),
    S("Grocery", "Baking Supplies", 45, 4.0, 0.7, peak=350, amp=0.5),
    S("Grocery", "Chips & Snacks", 10, 3.8, 1.3),
    S("Grocery", "Cookies & Candy", 12, 3.5, 1.1, peak=300, amp=0.3),
    S("Grocery", "Soft Drinks", 7, 5.5, peak=200, amp=0.3),
    S("Grocery", "Juice", 10, 4.0, 0.8),
    S("Grocery", "Bottled Water", 10, 4.5, 0.6, peak=200, amp=0.4),
    S("Grocery", "Frozen Meals", 10, 5.0, 1.1),
    S("Grocery", "Ice Cream", 14, 5.0, 0.9, peak=200, amp=0.5),
    S("Grocery", "Frozen Vegetables", 14, 2.5, 0.6),
    S("Grocery", "Fresh Meat", 7, 9.0),
    S("Grocery", "Seafood", 14, 11.0, 0.6),
    # Household
    S("Household", "Laundry Detergent", 30, 12.0),
    S("Household", "Dish Soap", 30, 3.5, 0.7),
    S("Household", "Paper Towels", 21, 9.0, 0.7),
    S("Household", "Toilet Paper", 21, 11.0, 0.8),
    S("Household", "Trash Bags", 45, 8.0, 0.6),
    S("Household", "Cleaning Supplies", 45, 5.0, 1.1),
    # Health & Beauty
    S("Health & Beauty", "Shampoo", 45, 6.0, 1.1),
    S("Health & Beauty", "Toothpaste", 40, 3.5, 0.8),
    S("Health & Beauty", "Body Wash", 40, 6.0, 0.9),
    S("Health & Beauty", "Vitamins", 60, 12.0),
    S("Health & Beauty", "Pain Relief", 90, 7.0, 0.7),
    S("Health & Beauty", "Skincare", 60, 14.0, 1.2),
    S("Health & Beauty", "Razors", 60, 12.0, 0.6),
    D("Health & Beauty", "Cosmetics", 11.0, 1.3),
    # Baby
    S("Baby", "Diapers", 10, 25.0, 0.8, requires="baby"),
    S("Baby", "Baby Formula", 10, 28.0, 0.6, requires="baby"),
    S("Baby", "Baby Food", 7, 6.0, 0.8, requires="baby"),
    D("Baby", "Baby Gear", 60.0, 0.9, requires="baby"),
    # Pet
    S("Pet", "Dog Food", 30, 25.0, requires="dog"),
    S("Pet", "Cat Food", 21, 15.0, requires="cat"),
    S("Pet", "Cat Litter", 30, 13.0, 0.6, requires="cat"),
    D("Pet", "Pet Toys", 8.0, 0.8),
    # Electronics
    D("Electronics", "Headphones", 45.0),
    D("Electronics", "Phone Accessories", 18.0, 1.2),
    D("Electronics", "TVs", 380.0, 0.8, peak=30, amp=0.4),
    D("Electronics", "Laptops", 550.0, 0.8, peak=220, amp=0.5),
    D("Electronics", "Video Games", 50.0, 1.1, peak=350, amp=0.6),
    D("Electronics", "Smart Home", 60.0, 0.8),
    S("Electronics", "Batteries", 90, 9.0, 0.6),
    # Clothing
    D("Clothing", "Men's Clothing", 20.0, 1.3),
    D("Clothing", "Women's Clothing", 22.0, 1.5),
    D("Clothing", "Kids' Clothing", 14.0, 1.1, peak=220, amp=0.6),
    D("Clothing", "Shoes", 35.0, 1.2),
    S("Clothing", "Socks & Underwear", 120, 10.0, 0.8),
    D("Clothing", "Outerwear", 55.0, 0.9, peak=330, amp=0.8),
    # Toys
    D("Toys", "Action Figures", 15.0, peak=350, amp=0.9),
    D("Toys", "Board Games", 22.0, 0.9, peak=350, amp=0.8),
    D("Toys", "Building Sets", 35.0, 0.9, peak=350, amp=0.9),
    D("Toys", "Dolls", 20.0, 0.8, peak=350, amp=0.8),
    D("Toys", "Outdoor Toys", 25.0, 0.8, peak=170, amp=0.8),
    # Home & Garden
    D("Home & Garden", "Bedding", 40.0),
    D("Home & Garden", "Kitchenware", 25.0, 1.3),
    D("Home & Garden", "Home Decor", 20.0, 1.3),
    D("Home & Garden", "Furniture", 150.0, 0.9),
    D("Home & Garden", "Storage", 18.0, 0.8),
    D("Home & Garden", "Garden Supplies", 15.0, 1.0, peak=120, amp=0.9),
    D("Home & Garden", "Grills & Outdoor", 180.0, 0.7, peak=160, amp=1.0),
    # Sports & Outdoors
    D("Sports & Outdoors", "Fitness Equipment", 45.0, peak=10, amp=0.6),
    D("Sports & Outdoors", "Camping", 60.0, 0.9, peak=180, amp=0.8),
    D("Sports & Outdoors", "Bikes", 200.0, 0.6, peak=130, amp=0.7),
    D("Sports & Outdoors", "Sports Balls", 18.0, 0.7),
    # Automotive
    S("Automotive", "Motor Oil", 120, 25.0, 0.6, requires="car"),
    D("Automotive", "Car Accessories", 20.0, 0.9, requires="car"),
    # Books & Media
    D("Books & Media", "Books", 14.0, 1.3),
    D("Books & Media", "Movies & Music", 15.0, 0.8),
    # Office & School
    D("Office & School", "School Supplies", 8.0, 1.0, peak=220, amp=1.0),
    S("Office & School", "Office Supplies", 60, 10.0, 0.8),
]

DEPARTMENTS: list[str] = list(dict.fromkeys(c.department for c in CATEGORIES))

AGE_BANDS = ("18-24", "25-34", "35-44", "45-54", "55-64", "65+")
REGIONS = ("Northeast", "Southeast", "Midwest", "Southwest", "West")


@dataclass(frozen=True)
class Segment:
    """A hidden shopper type. Individual shoppers are drawn around these means."""

    name: str
    share: float
    price_sensitivity: float  # higher = reacts more strongly to price
    quality_weight: float  # higher = pays more for quality
    store_brand_affinity: float  # >0 likes the store brand, <0 avoids it
    visits_per_week: float
    discretionary_per_visit: float  # average non-staple categories browsed per visit
    household_size: float
    p_dog: float
    p_cat: float
    p_baby: float
    p_car: float
    shop_hour: float  # typical hour of day they shop
    age_probs: tuple[float, ...]  # probabilities over AGE_BANDS
    dept_boost: dict[str, float] = field(default_factory=dict)  # interest per department (default 1)


SEGMENTS: list[Segment] = [
    Segment("budget_family", 0.25, 2.6, 0.6, 0.8, 1.6, 1.0, 4.2, 0.35, 0.25, 0.35, 0.85, 17,
            (0.02, 0.30, 0.40, 0.20, 0.06, 0.02),
            {"Grocery": 1.4, "Household": 1.3, "Baby": 1.5, "Toys": 1.6, "Clothing": 1.2,
             "Electronics": 0.7, "Home & Garden": 0.8}),
    Segment("young_professional", 0.18, 1.2, 1.4, -0.2, 0.9, 1.3, 1.5, 0.15, 0.25, 0.05, 0.70, 19,
            (0.20, 0.55, 0.20, 0.04, 0.01, 0.00),
            {"Electronics": 1.8, "Health & Beauty": 1.4, "Clothing": 1.3, "Sports & Outdoors": 1.3,
             "Toys": 0.3, "Grocery": 0.9}),
    Segment("premium_suburban", 0.15, 0.6, 2.0, -0.8, 1.1, 1.4, 3.2, 0.40, 0.20, 0.15, 0.95, 11,
            (0.00, 0.10, 0.35, 0.35, 0.15, 0.05),
            {"Home & Garden": 1.9, "Grocery": 1.2, "Sports & Outdoors": 1.3, "Electronics": 1.2}),
    Segment("student", 0.12, 3.0, 0.4, 0.5, 0.8, 0.9, 1.6, 0.03, 0.10, 0.01, 0.40, 20,
            (0.75, 0.22, 0.02, 0.01, 0.00, 0.00),
            {"Electronics": 1.4, "Books & Media": 1.5, "Office & School": 1.6, "Clothing": 1.2,
             "Home & Garden": 0.5, "Toys": 0.2}),
    Segment("retiree", 0.15, 1.8, 1.0, 0.2, 1.4, 0.8, 1.8, 0.25, 0.30, 0.00, 0.80, 10,
            (0.00, 0.00, 0.00, 0.05, 0.30, 0.65),
            {"Health & Beauty": 1.8, "Home & Garden": 1.4, "Books & Media": 1.4,
             "Electronics": 0.5, "Toys": 0.6, "Sports & Outdoors": 0.5}),
    Segment("active_outdoor", 0.15, 1.3, 1.2, 0.0, 1.0, 1.2, 2.5, 0.45, 0.10, 0.10, 0.90, 16,
            (0.10, 0.30, 0.30, 0.20, 0.08, 0.02),
            {"Sports & Outdoors": 2.2, "Automotive": 1.6, "Grocery": 1.0, "Books & Media": 0.6}),
]

# Brand-name building blocks. Each brand gets a unique prefix + suffix combination.
BRAND_PREFIXES = ("Bright", "North", "Silver", "Maple", "Oak", "Blue", "Summit", "Harbor", "Golden",
                  "Clear", "River", "Stone", "Pine", "Sun", "Iron", "Cedar", "Willow", "Frost",
                  "Coral", "Ember", "Aspen", "Lumen", "Prairie", "Crown")
BRAND_SUFFIXES = ("field", "lane", "crest", "wood", "ton", "vale", "ridge", "mark", "well", "haven",
                  "point", "brook")
STORE_BRAND = "Everyday Value"
PRODUCT_VARIANTS = ("Classic", "Original", "Premium", "Organic", "Family Size", "Lite", "Deluxe",
                    "Value Pack", "Essentials", "Select", "Signature", "Pro", "Plus", "Mini",
                    "Max", "Fresh", "Natural", "Ultra")
